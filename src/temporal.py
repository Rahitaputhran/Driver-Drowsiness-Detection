from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class TemporalState:
    closed_since: float | None = None
    yawn_since: float | None = None
    last_blink_time: float | None = None
    blink_count: int = 0
    yawn_count: int = 0


class TemporalFeatures:
    def __init__(self, window_seconds: float = 30.0):
        self.window_seconds = window_seconds
        self.samples = deque()
        self.state = TemporalState()

    def update(self, timestamp: float, ear: float, mar: float, ear_threshold: float, mar_threshold: float):
        closed = ear < ear_threshold
        yawning = mar > mar_threshold

        if closed and self.state.closed_since is None:
            self.state.closed_since = timestamp
        elif not closed and self.state.closed_since is not None:
            duration = timestamp - self.state.closed_since
            if duration >= 0.08:
                self.state.blink_count += 1
            self.state.last_blink_time = timestamp
            self.state.closed_since = None

        if yawning and self.state.yawn_since is None:
            self.state.yawn_since = timestamp
        elif not yawning and self.state.yawn_since is not None:
            if timestamp - self.state.yawn_since >= 0.7:
                self.state.yawn_count += 1
            self.state.yawn_since = None

        self.samples.append((timestamp, closed, yawning))
        cutoff = timestamp - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

        n = len(self.samples)
        closed_count = sum(1 for _, c, _ in self.samples if c)
        yawn_count = sum(1 for _, _, y in self.samples if y)

        perclos = closed_count / n if n else 0.0
        yawn_ratio = yawn_count / n if n else 0.0
        closed_duration = (
            timestamp - self.state.closed_since
            if self.state.closed_since is not None
            else 0.0
        )

        return {
            "perclos": perclos,
            "yawn_ratio": yawn_ratio,
            "eye_closed_duration": closed_duration,
            "blink_count": self.state.blink_count,
            "yawn_count": self.state.yawn_count,
            "window_samples": n,
        }
