from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainHistory:
    ''' Data class containing information about the current training iteration. '''

    iterations: int = 0
    time_millis: int = 0
    early_exit: bool = False
    scores: List[float] = field(default_factory=list)
