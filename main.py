"""
Agentic AI Smart Exam Surveillance System - Main Entry Point

This is the main entry point that initializes and runs the complete
surveillance pipeline with all agents working together.
"""

import argparse
import signal
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any

import cv2
import yaml
from loguru import logger

from agents.surveillance_agent import SurveillanceAgent, DemoSurveillanceAgent
from agents.detection_agent import DetectionAgent
from agents.tracking_agent import TrackingAgent
from agents.role_classification_agent import RoleClassificationAgent
from agents.behavior_analysis_agent import BehaviorAnalysisAgent
from agents.risk_scoring_agent import RiskScoringAgent
from agents.decision_agent import DecisionAgent


class ExamSurveillanceSystem:
    """
    Main system class that orchestrates all agents.
    
    Pipeline:
    1. Surveillance Agent captures frames
    2. Detection Agent detects persons and objects
    3. Tracking Agent tracks persons across frames
    4. Role Classification Agent identifies students vs invigilators
    5. Behavior Analysis Agent analyzes suspicious behaviors
    6. Risk Scoring Agent calculates risk scores
    7. Decision Agent triggers alerts
    """
    
    def __init__(self, config_path: str = "config/config.yaml", demo_mode: bool = False):
        self.config = self._load_config(config_path)
        self.demo_mode = demo_mode
        self.running = False
        self.frame_count = 0
        self.start_time = None
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize agents
        self._init_agents()
        
        # Alert callbacks
        self.alert_handlers = []
        
        logger.info("Exam Surveillance System initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "cameras": [
                {"id": "cam_1", "rtsp_url": "rtsp://admin:1234@192.168.1.120:554/stream", "name": "Main Camera", "enabled": True}
            ],
            "model": {
                "yolo_path": "models/yolo11n.pt",
                "device": "CPU"
            },
            "detection": {
                "confidence_threshold": 0.7,
                "input_size": [640, 640]
            },
            "tracking": {
                "max_age": 30,
                "min_hits": 3,
                "iou_threshold": 0.3
            },
            "risk": {
                "threshold": 70,
                "decay_rate": 5
            },
            "logging": {
                "level": "INFO"
            },
            "performance": {
                "max_fps": 18
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        level = log_config.get("level", "INFO")
        log_file = log_config.get("file", "logs/surveillance.log")
        
        # Create logs directory
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger.remove()
        logger.add(sys.stderr, level=level)
        logger.add(
            log_file,
            rotation=log_config.get("rotation", "10 MB"),
            retention=log_config.get("retention", "7 days"),
            level=level
        )
    
    def _init_agents(self):
        """Initialize all agents."""
        logger.info("Initializing agents...")
        
        # Surveillance Agent
        if self.demo_mode:
            self.config["demo"] = {"video_source": 0}
            self.surveillance_agent = DemoSurveillanceAgent(self.config)
        else:
            self.surveillance_agent = SurveillanceAgent(self.config)
        
        # Detection Agent
        self.detection_agent = DetectionAgent(self.config)
        
        # Tracking Agent
        self.tracking_agent = TrackingAgent(self.config)
        
        # Role Classification Agent
        self.role_agent = RoleClassificationAgent(self.config)
        
        # Behavior Analysis Agent
        self.behavior_agent = BehaviorAnalysisAgent(self.config)
        
        # Risk Scoring Agent
        self.risk_agent = RiskScoringAgent(self.config)
        
        # Decision Agent
        self.decision_agent = DecisionAgent(self.config)
        
        # Register alert callback
        self.decision_agent.register_callback(self._on_alert)
        
        logger.info("All agents initialized successfully")
    
    def _on_alert(self, decision):
        """Handle alert from decision agent."""
        logger.warning(f"ALERT: {decision.message}")
        
        for handler in self.alert_handlers:
            try:
                handler(decision)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def register_alert_handler(self, handler):
        """Register a custom alert handler."""
        self.alert_handlers.append(handler)
    
    def start(self):
        """Start the surveillance system."""
        logger.info("Starting Exam Surveillance System...")
        
        if not self.surveillance_agent.start():
            logger.error("Failed to start surveillance agent")
            return False
        
        self.running = True
        self.start_time = time.time()
        
        logger.info("System started successfully")
        return True
    
    def stop(self):
        """Stop the surveillance system."""
        logger.info("Stopping Exam Surveillance System...")
        self.running = False
        self.surveillance_agent.stop()
        logger.info("System stopped")
    
    def process_frame(self) -> Dict[str, Any]:
        """Process a single frame through the pipeline."""
        # Get frames from cameras
        frames = self.surveillance_agent.get_frames()
        
        if not frames:
            return {"status": "no_frames"}
        
        results = {}
        
        for camera_id, frame_data in frames.items():
            frame = frame_data.image
            self.frame_count += 1
            
            # 1. Detection
            detections = self.detection_agent.detect(frame)
            
            # 2. Tracking
            tracks = self.tracking_agent.update(detections)
            
            # 3. Role Classification
            roles = self.role_agent.classify(tracks)
            
            # 4. Filter to students only
            student_tracks = [t for t in tracks if self.role_agent.is_student(t.track_id)]
            
            # 5. Behavior Analysis
            behaviors = self.behavior_agent.analyze(frame, student_tracks)
            
            # 6. Associate objects to tracks
            associations = self.risk_agent.associate_detections_to_tracks(
                detections, student_tracks
            )
            
            # 7. Risk Scoring
            risk_scores = self.risk_agent.calculate_scores(
                detections, behaviors, associations
            )
            
            # 8. Decision Making
            decisions = self.decision_agent.decide(risk_scores)
            
            results[camera_id] = {
                "frame": frame,
                "detections": detections,
                "tracks": tracks,
                "roles": roles,
                "behaviors": behaviors,
                "risk_scores": risk_scores,
                "decisions": decisions
            }
        
        return results
    
    def run(self, display: bool = True):
        """Run the main processing loop."""
        if not self.start():
            return
        
        max_fps = self.config.get("performance", {}).get("max_fps", 18)
        frame_time = 1.0 / max_fps
        
        logger.info(f"Running at max {max_fps} FPS")
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Process frame
                results = self.process_frame()
                
                if results.get("status") == "no_frames":
                    time.sleep(0.01)
                    continue
                
                # Display if enabled
                if display:
                    for camera_id, data in results.items():
                        if "frame" in data:
                            display_frame = self._create_display(data)
                            cv2.imshow(f"Surveillance - {camera_id}", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self._reset_all()
                
                # FPS limiting
                elapsed = time.time() - loop_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
            if display:
                cv2.destroyAllWindows()
    
    def _create_display(self, data: Dict[str, Any]) -> Any:
        """Create display frame with overlays."""
        frame = data["frame"].copy()
        
        # Draw detections
        frame = self.detection_agent.draw_detections(frame, data["detections"])
        
        # Draw tracks
        frame = self.tracking_agent.draw_tracks(frame, data["tracks"])
        
        # Draw risk scores and alerts
        for track_id, risk in data.get("risk_scores", {}).items():
            decision = data.get("decisions", {}).get(track_id)
            
            if decision and decision.level.value in ["high", "critical"]:
                # Draw alert indicator
                track = self.tracking_agent.get_track(track_id)
                if track:
                    x1, y1, _, _ = track.bbox
                    cv2.putText(frame, f"ALERT: {risk.total_score:.0f}",
                               (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 0, 255), 2)
        
        # Draw FPS
        elapsed = time.time() - self.start_time if self.start_time else 1
        fps = self.frame_count / elapsed
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw stats
        stats = self.get_statistics()
        cv2.putText(frame, f"Students: {stats['students']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"High Risk: {stats['high_risk']}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def _reset_all(self):
        """Reset all agents."""
        logger.info("Resetting all agents...")
        self.tracking_agent.reset()
        self.role_agent.reset()
        self.behavior_agent.reset()
        self.risk_agent.reset()
        self.decision_agent.reset()
        self.frame_count = 0
        self.start_time = time.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        role_stats = self.role_agent.get_statistics()
        risk_stats = self.risk_agent.get_statistics()
        decision_stats = self.decision_agent.get_statistics()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            "running": self.running,
            "uptime": round(elapsed, 1),
            "frame_count": self.frame_count,
            "fps": round(self.frame_count / elapsed, 1) if elapsed > 0 else 0,
            "students": role_stats.get("students", 0),
            "invigilators": role_stats.get("invigilators", 0),
            "high_risk": risk_stats.get("high_risk", 0),
            "active_alerts": decision_stats.get("active_alerts", 0)
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic AI Smart Exam Surveillance System"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run in demo mode (use webcam)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display window"
    )
    
    args = parser.parse_args()
    
    # Create system
    system = ExamSurveillanceSystem(
        config_path=args.config,
        demo_mode=args.demo
    )
    
    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        system.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run
    system.run(display=not args.no_display)

if __name__ == "__main__":
    main()