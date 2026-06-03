from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimplificationSettings:
    importance_keep_threshold: float = 10.0
    unimportant_threshold: float = 10.0
    prune_rel_threshold: float = 0.1
    prune_abs_threshold: int = 5
    manual_protected_state_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class FrameworkSettings:
    seeds: tuple[int, ...] = (35, 123, 2025)
    min_state_count: int = 10
    n_clusters: int = 3
    use_filtered_step1_results: bool = False
    code_map: dict[str, int] | None = None
    simplification: SimplificationSettings = field(default_factory=SimplificationSettings)


# Baseline profile for real EHR data.
REAL_DATA_SETTINGS = FrameworkSettings(
    seeds=(35, 123, 2025),
    min_state_count=10,
    n_clusters=3,
    use_filtered_step1_results=False,
    simplification=SimplificationSettings(
        importance_keep_threshold=5.0,
        unimportant_threshold=5.0,
        prune_rel_threshold=0.1,
        prune_abs_threshold=3,
        manual_protected_state_ids=(),
    ),
)


# Separate profile for simulation runs.
SIMULATION_SETTINGS = FrameworkSettings(
    seeds=(35, 123, 2025),
    min_state_count=5,
    n_clusters=3,
    use_filtered_step1_results=False,
    code_map={
        "a": 0,
        "b": 10,
        "c": 30,
        "d": 54,
        "e": 166,  # discharge
        "f": 106,
        "g": 119,
        "h": 164,
        "i": 91,
        "u": 201,
        "v": 202,
        "w": 203,
        "t": 204,
        "x": 205,
        "y": 206,
        "z": 207,
    },
    simplification=SimplificationSettings(
        # Matches your simulation-focused thresholds.
        importance_keep_threshold=20.0,
        unimportant_threshold=20.0,
        prune_rel_threshold=0.15,
        prune_abs_threshold=3,
        manual_protected_state_ids=(91, 166),
    ),
)
