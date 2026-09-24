from dataclasses import dataclass


@dataclass
class CaptureState:
    phase: str = "idle"
    frames: int = 0
    reference_ready: bool = False
    repaired: bool = False
    return_to_region: bool = False
    model_revision: int = 0

    @property
    def stability(self) -> int:
        return 0 if self.phase == "idle" else 46 if self.phase == "failed" else 92 if self.phase == "passed" else min(88, 62 + self.frames * 4)

    @property
    def support(self) -> int:
        return 0 if self.phase == "idle" else 71 if self.phase == "failed" else 96 if self.phase == "passed" else min(93, self.frames * 16)
