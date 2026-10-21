from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    label: str
    score: float


class HysteresisDecision:
    """Prevents rapid state flipping near thresholds."""

    def __init__(self, warning_threshold=0.50, drowsy_threshold=0.75):
        self.warning_threshold = warning_threshold
        self.drowsy_threshold = drowsy_threshold
        self.state = "ALERT"

    def update(self, score: float) -> Decision:
        score = max(0.0, min(1.0, float(score)))

        if self.state == "DROWSY":
            if score < self.warning_threshold:
                self.state = "ALERT"
            elif score < self.drowsy_threshold:
                self.state = "WARNING"
        elif self.state == "WARNING":
            if score >= self.drowsy_threshold:
                self.state = "DROWSY"
            elif score < self.warning_threshold * 0.8:
                self.state = "ALERT"
        else:
            if score >= self.drowsy_threshold:
                self.state = "DROWSY"
            elif score >= self.warning_threshold:
                self.state = "WARNING"

        return Decision(self.state, score)
