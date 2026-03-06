"""
Surveillance Agent - RTSP Frame Capture

This agent handles:
- Connecting to CCTV cameras via RTSP
- Capturing and buffering frames
- Frame preprocessing (resize, normalize)
- Multi-camera support
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    id: str
    rtsp_url: str
    name: str
    enabled: bool = True


@dataclass
class Frame:
    """Captured frame with metadata."""
    image: np.ndarray
    camera_id: str
    timestamp: float
    frame_number: int


class CameraStream:
    """Handles a single camera RTSP stream."""
    
    def __init__(self, config: CameraConfig, buffer_size: int = 30):
        self.config = config
        self.buffer_size = buffer_size
        self.frame_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_count = 0
        self.fps = 0.0
        self.last_frame_time = 0.0
        
    def start(self) -> bool:
        """Start capturing frames from the camera."""
        if self.running:
            return True
            
        try:
            self.cap = cv2.VideoCapture(self.config.rtsp_url)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.config.id}: {self.config.rtsp_url}")
                return False
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"Started camera stream: {self.config.name} ({self.config.id})")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera {self.config.id}: {e}")
            return False
    
    def stop(self):
        """Stop capturing frames."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info(f"Stopped camera stream: {self.config.name}")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame from {self.config.id}, reconnecting...")
                time.sleep(1.0)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.config.rtsp_url)
                continue
            
            current_time = time.time()
            self.frame_count += 1
            fps_counter += 1
            
            # Calculate FPS every second
            if current_time - fps_start_time >= 1.0:
                self.fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            
            # Create frame object
            frame_obj = Frame(
                image=frame,
                camera_id=self.config.id,
                timestamp=current_time,
                frame_number=self.frame_count
            )
            
            # Add to buffer, drop oldest if full
            try:
                self.frame_buffer.put_nowait(frame_obj)
            except queue.Full:
                try:
                    self.frame_buffer.get_nowait()
                    self.frame_buffer.put_nowait(frame_obj)
                except queue.Empty:
                    pass
            
            self.last_frame_time = current_time
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Frame]:
        """Get the latest frame from the buffer."""
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest_frame(self) -> Optional[Frame]:
        """Get the most recent frame, discarding older ones."""
        latest = None
        while not self.frame_buffer.empty():
            try:
                latest = self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        return latest


class SurveillanceAgent:
    """Surveillance Agent for multi-camera RTSP capture."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cameras: Dict[str, CameraStream] = {}
        self.input_size = tuple(config.get("detection", {}).get("input_size", [640, 640]))
        self.running = False
        
        # Initialize cameras from config
        self._init_cameras()
        
    def _init_cameras(self):
        """Initialize camera streams from configuration."""
        camera_configs = self.config.get("cameras", [])
        
        for cam_config in camera_configs:
            if cam_config.get("enabled", True):
                camera = CameraConfig(
                    id=cam_config["id"],
                    rtsp_url=cam_config["rtsp_url"],
                    name=cam_config.get("name", cam_config["id"]),
                    enabled=cam_config.get("enabled", True)
                )
                self.cameras[camera.id] = CameraStream(camera)
                
        logger.info(f"Initialized {len(self.cameras)} camera(s)")
    
    def start(self) -> bool:
        """Start all camera streams."""
        success = True
        for camera_id, stream in self.cameras.items():
            if not stream.start():
                logger.error(f"Failed to start camera: {camera_id}")
                success = False
        
        self.running = success
        return success
    
    def stop(self):
        """Stop all camera streams."""
        self.running = False
        for stream in self.cameras.values():
            stream.stop()
        logger.info("Surveillance Agent stopped")
    
    def get_frames(self) -> Dict[str, Frame]:
        """Get latest frames from all cameras."""
        frames = {}
        for camera_id, stream in self.cameras.items():
            frame = stream.get_latest_frame()
            if frame:
                frames[camera_id] = frame
        return frames
    
    def get_frame(self, camera_id: str) -> Optional[Frame]:
        """Get latest frame from specific camera."""
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_latest_frame()
        return None
    
    def preprocess_frame(self, frame: Frame) -> np.ndarray:
        """Preprocess frame for detection model."""
        img = frame.image
        
        # Resize to model input size
        img = cv2.resize(img, self.input_size)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def add_camera(self, config: CameraConfig) -> bool:
        """Add a new camera at runtime."""
        if config.id in self.cameras:
            logger.warning(f"Camera {config.id} already exists")
            return False
        
        stream = CameraStream(config)
        if self.running:
            if not stream.start():
                return False
        
        self.cameras[config.id] = stream
        logger.info(f"Added camera: {config.name}")
        return True
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera at runtime."""
        if camera_id not in self.cameras:
            return False
        
        self.cameras[camera_id].stop()
        del self.cameras[camera_id]
        logger.info(f"Removed camera: {camera_id}")
        return True
    
    def get_camera_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all cameras."""
        status = {}
        for camera_id, stream in self.cameras.items():
            status[camera_id] = {
                "name": stream.config.name,
                "running": stream.running,
                "fps": round(stream.fps, 2),
                "frame_count": stream.frame_count,
                "buffer_size": stream.frame_buffer.qsize()
            }
        return status


# Demo mode for testing without RTSP
class DemoSurveillanceAgent(SurveillanceAgent):
    """Demo agent that generates synthetic frames for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cameras = {}
        self.input_size = tuple(config.get("detection", {}).get("input_size", [640, 640]))
        self.running = False
        self.frame_count = 0
        self.video_source = config.get("demo", {}).get("video_source", 0)
        self.cap: Optional[cv2.VideoCapture] = None
        
    def start(self) -> bool:
        """Start demo capture (webcam or video file)."""
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                logger.error(f"Failed to open demo source: {self.video_source}")
                return False
            self.running = True
            logger.info(f"Demo Surveillance Agent started with source: {self.video_source}")
            return True
        except Exception as e:
            logger.error(f"Error starting demo agent: {e}")
            return False
    
    def stop(self):
        """Stop demo capture."""
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("Demo Surveillance Agent stopped")
    
    def get_frames(self) -> Dict[str, Frame]:
        """Get frame from demo source."""
        if not self.cap or not self.running:
            return {}
        
        ret, image = self.cap.read()
        if not ret:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, image = self.cap.read()
            if not ret:
                return {}
        
        self.frame_count += 1
        frame = Frame(
            image=image,
            camera_id="demo_cam",
            timestamp=time.time(),
            frame_number=self.frame_count
        )
        
        return {"demo_cam": frame}


if __name__ == "__main__":
    # Test with webcam
    config = {
        "demo": {"video_source": 0},
        "detection": {"input_size": [640, 640]}
    }
    
    agent = DemoSurveillanceAgent(config)
    
    if agent.start():
        print("Press 'q' to quit")
        while True:
            frames = agent.get_frames()
            if "demo_cam" in frames:
                cv2.imshow("Demo", frames["demo_cam"].image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        agent.stop()
        cv2.destroyAllWindows()