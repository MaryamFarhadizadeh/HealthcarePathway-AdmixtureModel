import random
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass

# =========================
# Configuration
# =========================

N_PATIENTS = 400
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

START_STATE = "a"
END_STATE = "e"
SIMULATION_SCENARIO = "step1_medium"  # one of: step1_easy, step1_medium, step1_hard
BACKBONE_MODE = "three_source"  # one of: two_source, three_source

# Backbone pathways (ground truth).
# Important:
# - `two_source` keeps B1/B3 sharing the first post-start state ("b"),
#   so Step 1 root typically has 2 branches -> q has 2 components.
# - `three_source` gives each backbone a distinct first post-start state,
#   so Step 1 root can recover 3 branches -> q has 3 components.
BACKBONES_TWO_SOURCE = {
    "B1": ["a", "b", "f", "h", "e"],
    "B2": ["a", "c", "g", "h", "e"],
    "B3": ["a", "b", "d", "i", "e"],
}

BACKBONES_THREE_SOURCE = {
    "B1": ["a", "b", "f", "h", "e"],
    "B2": ["a", "c", "g", "h", "e"],
    "B3": ["a", "d", "i", "h", "e"],
}

if BACKBONE_MODE == "three_source":
    BACKBONES = BACKBONES_THREE_SOURCE
elif BACKBONE_MODE == "two_source":
    BACKBONES = BACKBONES_TWO_SOURCE
else:
    raise ValueError("BACKBONE_MODE must be 'two_source' or 'three_source'")

BACKBONE_WEIGHTS = {
    "B1": 0.45,
    "B2": 0.35,
    "B3": 0.20,
}

# Backbone-specific anchor codes to improve branch identifiability in Step 1.
# These are in CODE_MAP in settings_profiles.py.
ANCHOR_CODES = {
    "B1": "u",
    "B2": "v",
    "B3": "w",
}

# Keep noise separate from core pathway-defining states.
VERTICAL_NOISE_CODES = ["x", "y", "z"]
HORIZONTAL_NOISE_CODES = ["t"]


@dataclass(frozen=True)
class ScenarioConfig:
    p_add_vertical_noise: float
    max_vertical_noise: int
    p_add_horizontal_noise: float
    max_horizontal_nodes: int
    p_repeat_node: float
    p_skip_node: float
    p_add_anchor: float
    major_alpha: float
    minor_alpha: float


SCENARIOS: Dict[str, ScenarioConfig] = {
    # High signal, low noise: sanity-check scenario for Step 1 recovery.
    "step1_easy": ScenarioConfig(
        p_add_vertical_noise=0.20,
        max_vertical_noise=1,
        p_add_horizontal_noise=0.10,
        max_horizontal_nodes=1,
        p_repeat_node=0.05,
        p_skip_node=0.02,
        p_add_anchor=0.95,
        major_alpha=14.0,
        minor_alpha=0.8,
    ),
    # Balanced scenario for routine validation.
    "step1_medium": ScenarioConfig(
        p_add_vertical_noise=0.35,
        max_vertical_noise=1,
        p_add_horizontal_noise=0.20,
        max_horizontal_nodes=1,
        p_repeat_node=0.08,
        p_skip_node=0.05,
        p_add_anchor=0.75,
        major_alpha=8.0,
        minor_alpha=1.2,
    ),
    # Stress test scenario: expected to degrade branch separability.
    "step1_hard": ScenarioConfig(
        p_add_vertical_noise=0.55,
        max_vertical_noise=2,
        p_add_horizontal_noise=0.40,
        max_horizontal_nodes=2,
        p_repeat_node=0.15,
        p_skip_node=0.12,
        p_add_anchor=0.75,
        major_alpha=4.0,
        minor_alpha=2.0,
    ),
}

CFG = SCENARIOS[SIMULATION_SCENARIO]


# =========================
# Utilities
# =========================

def weighted_choice(weights: Dict[str, float]) -> str:
    r = random.random()
    cum = 0.0
    for k, w in weights.items():
        cum += w
        if r <= cum:
            return k
    return list(weights.keys())[-1]


# =========================
# Node construction
# =========================

def make_event_block(core_code: str) -> List[str]:
    # Start and end are atomic
    if core_code in (START_STATE, END_STATE):
        return [core_code]

    codes = {core_code}

    if random.random() < CFG.p_add_vertical_noise:
        n_extra = random.randint(0, CFG.max_vertical_noise)
        extras = random.sample(VERTICAL_NOISE_CODES, k=n_extra)
        codes.update(extras)

    return sorted(codes)


def insert_horizontal_noise(path: List[List[str]]) -> List[List[str]]:
    new_path = [path[0]]

    for prev_node, next_node in zip(path[:-1], path[1:]):
        # No noise after start or before end
        if prev_node != [START_STATE] and next_node != [END_STATE]:
            if random.random() < CFG.p_add_horizontal_noise:
                n_noise = random.randint(1, CFG.max_horizontal_nodes)
                for _ in range(n_noise):
                    noise_code = random.choice(HORIZONTAL_NOISE_CODES)
                    new_path.append([noise_code])

        new_path.append(next_node)

    return new_path


def remove_consecutive_duplicates(path: List[List[str]]) -> List[List[str]]:
    cleaned = [path[0]]
    for node in path[1:]:
        if set(node) != set(cleaned[-1]):
            cleaned.append(node)
    return cleaned


# =========================
# Patient simulation
# =========================

def simulate_single_patient(patient_id: str) -> Dict[str, Any]:
    dominant_backbone = weighted_choice(BACKBONE_WEIGHTS)
    backbone_keys = list(BACKBONES.keys())

    # Patient-level soft membership (ground-truth admixture proportions).
    alpha = [
        CFG.major_alpha if k == dominant_backbone else CFG.minor_alpha
        for k in backbone_keys
    ]
    gamma_draws = [random.gammavariate(a, 1.0) for a in alpha]
    gamma_sum = sum(gamma_draws)
    theta = [g / gamma_sum for g in gamma_draws]
    theta_map = {k: theta[i] for i, k in enumerate(backbone_keys)}

    # Build a mixed backbone by sampling source at each internal step.
    step_sources = []
    mixed_core = [START_STATE]
    n_steps = len(next(iter(BACKBONES.values())))
    for step_idx in range(1, n_steps - 1):
        src = weighted_choice(theta_map)
        step_sources.append(src)
        mixed_core.append(BACKBONES[src][step_idx])
    mixed_core.append(END_STATE)

    expanded = []

    for state in mixed_core:
        # Skip only non-terminal states
        if state not in (START_STATE, END_STATE):
            if random.random() < CFG.p_skip_node:
                continue

        expanded.append(state)

        # Repeat only non-terminal states
        if state not in (START_STATE, END_STATE):
            if random.random() < CFG.p_repeat_node:
                expanded.append(state)

    # Convert to event blocks
    blocks = [make_event_block(s) for s in expanded]

    # Add horizontal noise
    blocks = insert_horizontal_noise(blocks)

    # Add branch-specific anchor before discharge with high probability.
    if random.random() < CFG.p_add_anchor:
        # Use dominant backbone anchor (not per-step source) as latent class marker.
        anchor = ANCHOR_CODES[dominant_backbone]
        if blocks[-1] == [END_STATE]:
            blocks.insert(-1, [anchor])
        else:
            blocks.append([anchor])

    # Remove consecutive duplicates
    blocks = remove_consecutive_duplicates(blocks)

    # Enforce correct start/end
    if blocks[0] != [START_STATE]:
        blocks.insert(0, [START_STATE])
    if blocks[-1] != [END_STATE]:
        blocks.append([END_STATE])

    return {
        "patient_id": patient_id,
        "backbone": dominant_backbone,
        "trajectory": blocks,
        "theta_B1": theta_map.get("B1", 0.0),
        "theta_B2": theta_map.get("B2", 0.0),
        "theta_B3": theta_map.get("B3", 0.0),
        "n_source_switches": sum(
            1 for i in range(1, len(step_sources)) if step_sources[i] != step_sources[i - 1]
        ),
    }


def simulate_cohort(n_patients: int) -> List[Dict[str, Any]]:
    cohort = []
    for i in range(1, n_patients + 1):
        pid = f"P{i:03d}"
        cohort.append(simulate_single_patient(pid))
    return cohort


# =========================
# CSV export
# =========================

def save_event_level_csv(cohort, filename="simulated_events.csv"):
    rows = []
    for patient in cohort:
        t = 0
        for block in patient["trajectory"]:
            for code in block:
                rows.append({
                    "patient_id": patient["patient_id"],
                    "time": t,
                    "code": code,
                    "backbone": patient["backbone"]
                })
            t += 1

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df


def save_patient_level_csv(cohort, filename="simulated_patients.csv"):
    rows = []
    for patient in cohort:
        traj_str = " -> ".join(
            ["{" + ",".join(block) + "}" for block in patient["trajectory"]]
        )
        rows.append({
            "patient_id": patient["patient_id"],
            "backbone": patient["backbone"],
            "trajectory": traj_str,
            "length": len(patient["trajectory"]),
            "theta_B1": patient.get("theta_B1", None),
            "theta_B2": patient.get("theta_B2", None),
            "theta_B3": patient.get("theta_B3", None),
            "n_source_switches": patient.get("n_source_switches", None),
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df


# =========================
# Run simulation
# =========================

if __name__ == "__main__":
    print(f"Scenario: {SIMULATION_SCENARIO}")
    print(f"Backbone mode: {BACKBONE_MODE}")
    print(f"Config: {CFG}")
    cohort = simulate_cohort(N_PATIENTS)

    df_events = save_event_level_csv(cohort, "simulated_events.csv")
    df_patients = save_patient_level_csv(cohort, "simulated_patients.csv")

    print("Saved:")
    print(" - simulated_events.csv")
    print(" - simulated_patients.csv")
    print("\nEvent-level preview:")
    print(df_events.head())
