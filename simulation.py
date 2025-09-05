## Very unfinished

import numpy as np
from dataclasses import dataclass

@dataclass
class SimResult:
    calc_rank_analytic: np.ndarray
    calc_rank: np.ndarray
    f_j_k: float
    