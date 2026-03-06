"""
Decision Agent - Alert Decision Making

This agent handles:
- Deciding when to trigger alerts based on risk scores
- Managing alert states
- Coordinating with the alert system
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from loguru import logger

from agents.risk_scoring_agent import RiskScore


class AlertLevel(Enum):
    """Alert severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertAction(Enum):
    """Actions to take on alert."""
    CONTINUE_MONITORING = "continue_monitoring"
    LOG_EVENT = "log_event"
    CAPTURE_EVIDENCE = "capture_evidence"
    TRIGGER_WARNING = "trigger_warning"
    TRIGGER_FULL_ALERT = "trigger_full_alert"


@dataclass
class AlertDecision:
    """Decision made by the agent."""
    track_id: int
    level: AlertLevel
    actions: List[AlertAction]
    risk_score: float
    timestamp: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class DecisionAgent:
    """
    Decision Agent for alert triggering.
    
    Decision Logic:
    - Risk < 30: Continue monitoring
    - Risk 30-50: Log event
    - Risk 50-70: Capture evidence
    - Risk 70-85: Trigger warning
    - Risk >= 85: Full alert
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        risk_config = config.get("risk", {})
        
        self.threshold = risk_config.get("threshold", 70)
        
        # Alert thresholds
        self.thresholds = {
            AlertLevel.LOW: 30,
            AlertLevel.MEDIUM: 50,
            AlertLevel.HIGH: 70,
            AlertLevel.CRITICAL: 85
        }
        
        # Track alert states
        self.alert_states: Dict[int, AlertDecision] = {}
        self.alert_history: Dict[int, List[AlertDecision]] = defaultdict(list)
        self.cooldown_tracker: Dict[int, float] = {}
        
        # Cooldown period between alerts for same track
        self.alert_cooldown = 30.0  # seconds
        
        # Callbacks
        self.alert_callbacks: List[Callable[[AlertDecision], None]] = []
        
        logger.info("Decision Agent initialized")
    
    def decide(self, risk_scores: Dict[int, RiskScore]) -> Dict[int, AlertDecision]:
        """
        Make decisions for all tracked persons.
        
        Args:
            risk_scores: Dictionary of track_id to RiskScore
            
        Returns:
            Dictionary of track_id to AlertDecision
        """
        current_time = time.time()
        decisions = {}
        
        for track_id, risk in risk_scores.items():
            decision = self._make_decision(track_id, risk, current_time)
            decisions[track_id] = decision
            
            # Store in history
            self.alert_states[track_id] = decision
            
            if decision.level != AlertLevel.NONE:
                self.alert_history[track_id].append(decision)
                
                # Execute callbacks
                self._execute_callbacks(decision)
        
        return decisions
    
    def _make_decision(self, track_id: int, risk: RiskScore, current_time: float) -> AlertDecision:
        """Make decision for a single tracked person."""
        score = risk.total_score
        
        # Determine alert level
        level = AlertLevel.NONE
        for alert_level, threshold in sorted(self.thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                level = alert_level
                break
        
        # Determine actions based on level
        actions = self._determine_actions(level, track_id, current_time)
        
        # Generate message
        message = self._generate_message(level, score, risk)
        
        return AlertDecision(
            track_id=track_id,
            level=level,
            actions=actions,
            risk_score=score,
            timestamp=current_time,
            message=message,
            details={
                "breakdown": risk.breakdown,
                "event_count": len(risk.events)
            }
        )
    
    def _determine_actions(self, level: AlertLevel, track_id: int, current_time: float) -> List[AlertAction]:
        """Determine actions based on alert level."""
        actions = []
        
        # Check cooldown
        if track_id in self.cooldown_tracker:
            if current_time - self.cooldown_tracker[track_id] < self.alert_cooldown:
                # In cooldown, only continue monitoring
                return [AlertAction.CONTINUE_MONITORING]
        
        if level == AlertLevel.NONE:
            actions.append(AlertAction.CONTINUE_MONITORING)
        
        elif level == AlertLevel.LOW:
            actions.append(AlertAction.LOG_EVENT)
            actions.append(AlertAction.CONTINUE_MONITORING)
        
        elif level == AlertLevel.MEDIUM:
            actions.append(AlertAction.LOG_EVENT)
            actions.append(AlertAction.CAPTURE_EVIDENCE)
            actions.append(AlertAction.CONTINUE_MONITORING)
        
        elif level == AlertLevel.HIGH:
            actions.append(AlertAction.LOG_EVENT)
            actions.append(AlertAction.CAPTURE_EVIDENCE)
            actions.append(AlertAction.TRIGGER_WARNING)
            self.cooldown_tracker[track_id] = current_time
        
        elif level == AlertLevel.CRITICAL:
            actions.append(AlertAction.LOG_EVENT)
            actions.append(AlertAction.CAPTURE_EVIDENCE)
            actions.append(AlertAction.TRIGGER_FULL_ALERT)
            self.cooldown_tracker[track_id] = current_time
        
        return actions
    
    def _generate_message(self, level: AlertLevel, score: float, risk: RiskScore) -> str:
        """Generate alert message."""
        if level == AlertLevel.NONE:
            return f"Track {risk.track_id}: Normal behavior (score: {score:.1f})"
        
        elif level == AlertLevel.LOW:
            return f"Track {risk.track_id}: Minor suspicious activity detected (score: {score:.1f})"
        
        elif level == AlertLevel.MEDIUM:
            return f"Track {risk.track_id}: Suspicious behavior - capturing evidence (score: {score:.1f})"
        
        elif level == AlertLevel.HIGH:
            behaviors = ", ".join(risk.breakdown.keys()) if risk.breakdown else "multiple events"
            return f"WARNING - Track {risk.track_id}: High risk behavior detected ({behaviors}) - Score: {score:.1f}"
        
        elif level == AlertLevel.CRITICAL:
            behaviors = ", ".join(risk.breakdown.keys()) if risk.breakdown else "multiple events"
            return f"ALERT - Track {risk.track_id}: Critical malpractice detected ({behaviors}) - Score: {score:.1f}"
        
        return f"Track {risk.track_id}: Unknown state"
    
    def register_callback(self, callback: Callable[[AlertDecision], None]):
        """Register a callback for alert events."""
        self.alert_callbacks.append(callback)
    
    def _execute_callbacks(self, decision: AlertDecision):
        """Execute registered callbacks for an alert."""
        if decision.level in [AlertLevel.HIGH, AlertLevel.CRITICAL]:
            for callback in self.alert_callbacks:
                try:
                    callback(decision)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    def get_active_alerts(self) -> List[AlertDecision]:
        """Get all currently active alerts."""
        return [
            decision for decision in self.alert_states.values()
            if decision.level in [AlertLevel.HIGH, AlertLevel.CRITICAL]
        ]
    
    def get_alert_history(self, track_id: int) -> List[AlertDecision]:
        """Get alert history for a specific track."""
        return self.alert_history.get(track_id, [])
    
    def get_decision(self, track_id: int) -> Optional[AlertDecision]:
        """Get current decision for a track."""
        return self.alert_states.get(track_id)
    
    def acknowledge_alert(self, track_id: int):
        """Acknowledge an alert (resets cooldown)."""
        self.cooldown_tracker.pop(track_id, None)
        logger.info(f"Alert acknowledged for track {track_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decision statistics."""
        if not self.alert_states:
            return {
                "total_decisions": 0,
                "active_alerts": 0,
                "level_counts": {}
            }
        
        level_counts = defaultdict(int)
        for decision in self.alert_states.values():
            level_counts[decision.level.value] += 1
        
        return {
            "total_decisions": len(self.alert_states),
            "active_alerts": len(self.get_active_alerts()),
            "level_counts": dict(level_counts)
        }
    
    def reset(self):
        """Reset all decision data."""
        self.alert_states.clear()
        self.alert_history.clear()
        self.cooldown_tracker.clear()


if __name__ == "__main__":
    config = {
        "risk": {
            "threshold": 70
        }
    }
    
    agent = DecisionAgent(config)
    
    # Test with mock risk score
    from dataclasses import dataclass
    
    @dataclass
    class MockRiskScore:
        track_id: int
        total_score: float
        breakdown: dict
        events: list
        is_alert_triggered: bool
    
    # Low risk
    risk1 = MockRiskScore(1, 25.0, {}, [], False)
    
    # High risk
    risk2 = MockRiskScore(2, 75.0, {"phone_usage": 60, "head_turning": 15}, [], True)
    
    decisions = agent.decide({1: risk1, 2: risk2})
    
    for track_id, decision in decisions.items():
        print(f"Track {track_id}: {decision.level.value} - {decision.message}")
