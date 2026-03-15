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

SIMULATION_SCENARIO = "step1_hard"  # step1_easy, step1_medium, step1_hard
BACKBONE_MODE = "three_source"  # two_source, three_source


# =========================
# Backbone Definitions
# =========================

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

BACKBONES = (
    BACKBONES_THREE_SOURCE if BACKBONE_MODE == "three_source"
    else BACKBONES_TWO_SOURCE
)

# Backbone prevalence in population
BACKBONE_WEIGHTS = {
    "B1": 0.45,
    "B2": 0.35,
    "B3": 0.20,
}

# Anchor codes (optional signal)
ANCHOR_CODES = {
    "B1": "u",
    "B2": "v",
    "B3": "w",
}

VERTICAL_NOISE_CODES = ["x", "y", "z"]
HORIZONTAL_NOISE_CODES = ["t"]


# =========================
# Scenario Configuration
# =========================

@dataclass(frozen=True)
class ScenarioConfig:
    p_add_vertical_noise: float
    max_vertical_noise: int
    p_add_horizontal_noise: float
    max_horizontal_nodes: int
    p_repeat_node: float
    p_skip_node: float
    p_add_anchor: float
    dirichlet_alpha: float  # controls sparsity of mixture
    p_backbone_switch: float  # probability to switch source at each inner step


SCENARIOS: Dict[str, ScenarioConfig] = {

    "step1_easy": ScenarioConfig(
        0.20, 1,
        0.10, 1,
        0.05, 0.02,
        0.95,
        0.8,
        0.10,
    ),

    "step1_medium": ScenarioConfig(
        0.35, 1,
        0.20, 1,
        0.08, 0.05,
        0.75,
        1.0,
        0.22,
    ),

    "step1_hard": ScenarioConfig(
        0.55, 2,
        0.25, 2,
        0.10, 0.06,
        0.75,
        1.2,
        0.35,
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


def dirichlet_sample(alpha_value: float, keys: List[str]) -> Dict[str, float]:
    gamma_draws = [random.gammavariate(alpha_value, 1.0) for _ in keys]
    total = sum(gamma_draws)
    return {k: gamma_draws[i] / total for i, k in enumerate(keys)}


# =========================
# Event Block Construction
# =========================

def make_event_block(core_code: str) -> List[str]:
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
# Patient Simulation (Option A)
# =========================

def simulate_single_patient(patient_id: str) -> Dict[str, Any]:

    backbone_keys = list(BACKBONES.keys())

    # Step 1: Sample Dirichlet mixture
    theta_map = dirichlet_sample(CFG.dirichlet_alpha, backbone_keys)

    # Step 2: Keep a dominant label for evaluation/reporting.
    latent_backbone = max(theta_map.items(), key=lambda x: x[1])[0]

    # Step 3: Generate a mostly dominant trajectory with controlled switching.
    backbone_len = min(len(BACKBONES[k]) for k in backbone_keys)
    base_traj = [START_STATE]
    for pos in range(1, backbone_len - 1):
        src = latent_backbone
        if random.random() < CFG.p_backbone_switch:
            src = weighted_choice(theta_map)
        base_traj.append(BACKBONES[src][pos])
    base_traj.append(END_STATE)

    # Step 4: Apply skip / repeat noise
    expanded = []
    for state in base_traj:

        if state not in (START_STATE, END_STATE):
            if random.random() < CFG.p_skip_node:
                continue

        expanded.append(state)

        if state not in (START_STATE, END_STATE):
            if random.random() < CFG.p_repeat_node:
                expanded.append(state)

    # Step 5: Convert to event blocks
    blocks = [make_event_block(s) for s in expanded]

    # Step 6: Horizontal noise
    blocks = insert_horizontal_noise(blocks)

    # Step 7: Optional anchor (linked to dominant source, not per-step source)
    if random.random() < CFG.p_add_anchor:
        anchor = ANCHOR_CODES[latent_backbone]
        if blocks[-1] == [END_STATE]:
            blocks.insert(-1, [anchor])
        else:
            blocks.append([anchor])

    blocks = remove_consecutive_duplicates(blocks)

    if blocks[0] != [START_STATE]:
        blocks.insert(0, [START_STATE])
    if blocks[-1] != [END_STATE]:
        blocks.append([END_STATE])

    return {
        "patient_id": patient_id,
        "latent_backbone": latent_backbone,
        "theta_B1": theta_map.get("B1", 0.0),
        "theta_B2": theta_map.get("B2", 0.0),
        "theta_B3": theta_map.get("B3", 0.0),
        "trajectory": blocks,
    }


def simulate_cohort(n_patients: int):
    return [
        simulate_single_patient(f"P{i:03d}")
        for i in range(1, n_patients + 1)
    ]


# =========================
# CSV Export
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
                    "latent_backbone": patient["latent_backbone"]
                })
            t += 1
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df


def save_patient_level_csv(cohort, filename="simulated_patients.csv"):
    rows = []
    for patient in cohort:
        rows.append({
            "patient_id": patient["patient_id"],
            "latent_backbone": patient["latent_backbone"],
            "theta_B1": patient["theta_B1"],
            "theta_B2": patient["theta_B2"],
            "theta_B3": patient["theta_B3"],
            "length": len(patient["trajectory"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df


# =========================
# Run
# =========================

if __name__ == "__main__":
    print(f"Scenario: {SIMULATION_SCENARIO}")
    print(f"Backbone mode: {BACKBONE_MODE}")
    print(f"Dirichlet alpha: {CFG.dirichlet_alpha}")

    cohort = simulate_cohort(N_PATIENTS)

    save_event_level_csv(cohort)
    save_patient_level_csv(cohort)

    print("Simulation completed successfully.")
