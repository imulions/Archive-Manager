#!/usr/bin/env python3
"""Minimal data-free reproduction of the rule-based archive comparison.

The default run generates deterministic synthetic planar trajectories.  No
paper data, checkpoints, or physical-device interfaces are required.  The
same quality rule, fixed GMM occupancy descriptor, redundancy threshold, and
archive-selection rules can also be applied to user-provided state sequences
through the optional NPZ input format described in README.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# Locked protocol values corresponding to the manuscript's direct evaluator.
POSITION_THRESHOLD = 0.15
YAW_THRESHOLD = 0.15
REACH_HORIZON = 500
FINAL_WINDOW_FRACTION = 0.20
GMM_COMPONENTS = 12
DELTA_QUANTILE = 0.30
ARCHIVE_BUDGET = 8
SYNTHETIC_TRAJECTORY_STEPS = 200
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
SYNTHETIC_COUNTS = {
    "demonstrations": 32,
    "calibration": 48,
    "candidate": 240,
    "heldout": 26,
}
RECORDED_COUNTS = {
    "demonstrations": 17,
    "calibration": 48,
    "candidate": 240,
    "heldout": 26,
}
METHODS = ("uniform", "quality_only", "novelty_only", "quality_novelty")
METHOD_LABELS = {
    "uniform": "Uniform",
    "quality_only": "Quality-only",
    "novelty_only": "Novelty-only",
    "quality_novelty": "Q + q + d",
}
GOAL = np.array([1.0, 0.0], dtype=float)
ACTION_LOW = np.array([-0.24, -0.18, -0.35], dtype=float)
ACTION_HIGH = np.array([0.28, 0.18, 0.35], dtype=float)


@dataclass
class Trajectory:
    """State-only trajectory record used by the public archive protocol."""

    states: np.ndarray
    source: str = ""
    descriptor: np.ndarray | None = None
    quality: float = 0.0
    feasible: int = 0
    metrics: dict[str, float | int] | None = None


@dataclass(frozen=True)
class Mode:
    """A simple controllable trajectory-generation mode."""

    name: str
    lateral_offset: float
    speed: float
    yaw_gain: float
    yaw_bias: float
    noise_scale: float


MODES = (
    Mode("near_left", -0.06, 1.00, 2.00, 0.00, 0.65),
    Mode("near_right", 0.06, 1.00, 2.00, 0.00, 0.65),
    Mode("left_arc", -0.20, 1.00, 2.10, 0.00, 0.85),
    Mode("right_arc", 0.20, 1.00, 2.10, 0.00, 0.85),
    Mode("slow", 0.00, 0.78, 1.55, 0.00, 0.55),
    Mode("fast", 0.00, 1.22, 2.45, 0.00, 0.75),
    Mode("yaw_bias", 0.00, 0.98, 2.00, 0.05, 0.95),
    Mode("unstable", 0.42, 1.35, 4.20, 0.20, 1.35),
)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to [-pi, pi)."""

    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def generate_trajectory(
    mode: Mode,
    goal: np.ndarray,
    seed: int,
    *,
    initial_xy_std: float,
    initial_yaw_std: float,
    observation_noise: float,
    disturbance_std: float,
    dynamics_scale: float,
    steps: int = SYNTHETIC_TRAJECTORY_STEPS,
) -> Trajectory:
    """Generate one synthetic planar rollout used by the evaluator.

    This is a compact surrogate for the state/action collection process.  It
    creates observable [x, y, yaw] states with controlled variation in initial
    condition, noise, disturbance, and path shape.
    """

    rng = np.random.default_rng(seed)
    state = np.array(
        [
            -1.25 + rng.normal(0.0, initial_xy_std),
            rng.normal(0.0, initial_xy_std),
            rng.normal(0.0, initial_yaw_std),
        ],
        dtype=float,
    )
    velocity = np.zeros(3, dtype=float)
    states: list[np.ndarray] = []

    for _ in range(steps):
        observation = state + rng.normal(0.0, observation_noise, size=3)
        displacement = goal - observation[:2]
        distance = float(np.linalg.norm(displacement))
        decay = math.exp(-distance / 0.85)
        perpendicular = np.array([-displacement[1], displacement[0]])
        virtual_goal = goal + perpendicular * mode.lateral_offset * (1.0 - decay)
        relative = virtual_goal - observation[:2]
        desired_yaw = math.atan2(relative[1], relative[0])
        heading_error = float(wrap_angle(desired_yaw - observation[2]))

        surge = mode.speed * distance * max(0.12, math.cos(heading_error))
        sway = 0.95 * distance * math.sin(heading_error)
        sway += 0.50 * mode.lateral_offset * (1.0 - decay)
        yaw_force = mode.yaw_gain * heading_error + mode.yaw_bias
        action = np.array([surge, sway, yaw_force]) * 0.105
        action += rng.normal(
            0.0, observation_noise * mode.noise_scale, size=3
        )
        action = np.clip(action, ACTION_LOW, ACTION_HIGH)

        mass = np.array([1.00, 1.15, 0.42]) * dynamics_scale
        damping = np.array([1.10, 1.20, 0.72])
        gain = np.array([3.2, 3.1, 1.9])
        disturbance = rng.normal(0.0, disturbance_std, size=3)
        acceleration = (gain * action - damping * velocity) / mass + disturbance
        velocity = velocity + 0.10 * acceleration
        rotation = np.array(
            [
                [math.cos(state[2]), -math.sin(state[2])],
                [math.sin(state[2]), math.cos(state[2])],
            ]
        )
        state = state.copy()
        state[:2] += 0.10 * (rotation @ velocity[:2])
        state[2] = float(wrap_angle(state[2] + 0.10 * velocity[2]))
        states.append(state.copy())

    return Trajectory(states=np.asarray(states), source=mode.name)


def quality_pair(
    states: np.ndarray,
    goal: np.ndarray,
) -> tuple[float, int, dict[str, float | int]]:
    """Compute the paper's direct pair (Q, q) from observable states."""

    states = np.asarray(states, dtype=float)
    goal = np.asarray(goal, dtype=float).reshape(2)
    position_error = np.linalg.norm(states[:, :2] - goal.reshape(1, 2), axis=1)
    tail_start = max(
        0, len(states) - int(np.ceil(len(states) * FINAL_WINDOW_FRACTION))
    )
    tail = states[tail_start:]
    terminal_error = float(
        np.mean(np.linalg.norm(tail[:, :2] - goal.reshape(1, 2), axis=1))
    )
    mean_yaw = float(
        np.arctan2(np.mean(np.sin(tail[:, 2])), np.mean(np.cos(tail[:, 2])))
    )
    terminal_yaw = float(
        np.sqrt(np.mean(wrap_angle(tail[:, 2] - mean_yaw) ** 2))
    )
    reached = np.flatnonzero(position_error <= POSITION_THRESHOLD)
    first_reach = int(reached[0] + 1) if len(reached) else REACH_HORIZON + 1
    margins = {
        "position_margin": 1.0 - terminal_error / POSITION_THRESHOLD,
        "yaw_margin": 1.0 - terminal_yaw / YAW_THRESHOLD,
        "time_margin": 1.0 - first_reach / REACH_HORIZON,
    }
    quality = float(min(margins.values()))
    feasible = int(
        terminal_error <= POSITION_THRESHOLD
        and terminal_yaw <= YAW_THRESHOLD
        and first_reach <= REACH_HORIZON
    )
    return quality, feasible, {
        "terminal_position_error": terminal_error,
        "terminal_yaw_stability": terminal_yaw,
        "first_reach": first_reach,
        **margins,
    }


class StateVisitationEncoder:
    """Fixed full-covariance GMM occupancy encoder."""

    def __init__(
        self,
        components: int = GMM_COMPONENTS,
        seed: int = 0,
        n_init: int = 1,
    ) -> None:
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=components,
            covariance_type="full",
            reg_covar=1e-5,
            n_init=n_init,
            random_state=seed,
        )

    @staticmethod
    def features(states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        yaw = states[:, 2]
        return np.column_stack(
            [states[:, 0], states[:, 1], np.sin(yaw), np.cos(yaw)]
        )

    def fit(self, trajectories: Sequence[Trajectory]) -> None:
        pooled = np.vstack([self.features(item.states) for item in trajectories])
        self.gmm.fit(self.scaler.fit_transform(pooled))

    def encode(self, states: np.ndarray) -> np.ndarray:
        posterior = self.gmm.predict_proba(
            self.scaler.transform(self.features(states))
        )
        descriptor = posterior.mean(axis=0)
        total = float(descriptor.sum())
        return descriptor / total if total > 1e-12 else descriptor


def annotate(
    trajectories: Iterable[Trajectory],
    encoder: StateVisitationEncoder,
    goal: np.ndarray,
) -> list[Trajectory]:
    """Evaluate and encode trajectories with one frozen protocol."""

    result = []
    for item in trajectories:
        quality, feasible, metrics = quality_pair(item.states, goal)
        result.append(
            Trajectory(
                states=item.states,
                source=item.source,
                descriptor=encoder.encode(item.states),
                quality=quality,
                feasible=feasible,
                metrics=metrics,
            )
        )
    return result


def nearest_distances(descriptors: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(
        descriptors[:, None, :] - descriptors[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def archive_distance_matrix(archive: Sequence[Trajectory]) -> np.ndarray:
    descriptors = np.vstack([item.descriptor for item in archive])
    return np.linalg.norm(
        descriptors[:, None, :] - descriptors[None, :, :], axis=2
    )


def select_archive(
    candidates: Sequence[Trajectory],
    method: str,
    budget: int,
    delta: float,
    seed: int,
) -> list[Trajectory]:
    """Apply the four fixed-pool archive rules."""

    if not candidates:
        return []
    budget = min(budget, len(candidates))
    if method == "uniform":
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(candidates), size=budget, replace=False)
        return [candidates[int(index)] for index in indices]

    if method == "quality_only":
        ordered = sorted(
            candidates,
            key=lambda item: (item.feasible, item.quality),
            reverse=True,
        )
        return ordered[:budget]

    descriptors = np.vstack([item.descriptor for item in candidates])
    if method == "novelty_only":
        centroid = descriptors.mean(axis=0)
        chosen = [int(np.argmin(np.linalg.norm(descriptors - centroid, axis=1)))]
        while len(chosen) < budget:
            remaining = [i for i in range(len(candidates)) if i not in chosen]
            nearest = np.min(
                np.linalg.norm(
                    descriptors[remaining, None, :]
                    - descriptors[chosen][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            chosen.append(remaining[int(np.argmax(nearest))])
        return [candidates[index] for index in chosen]

    if method != "quality_novelty":
        raise ValueError(f"Unknown archive rule: {method}")

    feasible = sorted(
        [item for item in candidates if item.feasible == 1],
        key=lambda item: item.quality,
        reverse=True,
    )
    if not feasible:
        return []
    archive: list[Trajectory] = []
    for candidate in feasible:
        if not archive:
            archive.append(candidate)
            continue
        archive_descriptors = np.vstack([item.descriptor for item in archive])
        distances = np.linalg.norm(
            archive_descriptors - candidate.descriptor.reshape(1, -1),
            axis=1,
        )
        nearest_index = int(np.argmin(distances))
        if distances[nearest_index] > delta and len(archive) < budget:
            archive.append(candidate)
        elif candidate.quality > archive[nearest_index].quality:
            archive[nearest_index] = candidate

    # Match the fixed-pool protocol: fill remaining capacity using feasible
    # candidates and farthest-point selection, after the quality/novelty pass.
    archive_ids = {id(item) for item in archive}
    remaining = [item for item in feasible if id(item) not in archive_ids]
    while remaining and len(archive) < budget:
        archive_descriptors = np.vstack([item.descriptor for item in archive])
        farthest = [
            float(
                np.min(
                    np.linalg.norm(
                        item.descriptor.reshape(1, -1) - archive_descriptors,
                        axis=1,
                    )
                )
            )
            for item in remaining
        ]
        index = int(np.argmax(farthest))
        archive.append(remaining.pop(index))
    return archive[:budget]


def archive_metrics(
    archive: Sequence[Trajectory],
    support: np.ndarray,
    delta: float,
) -> dict[str, float | int]:
    if not archive:
        return {
            "archive_size": 0,
            "redundancy_ratio": float("nan"),
            "mean_nearest_distance": float("nan"),
            "descriptor_clusters": 0,
            "coverage_radius": float("nan"),
            "archive_mean_Q": float("nan"),
            "archive_feasible_ratio": float("nan"),
        }
    distances = archive_distance_matrix(archive)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1) if len(archive) > 1 else np.array([np.nan])
    descriptors = np.vstack([item.descriptor for item in archive])
    support_distance = np.linalg.norm(
        support[:, None, :] - descriptors[None, :, :], axis=2
    ).min(axis=1)
    return {
        "archive_size": len(archive),
        "redundancy_ratio": float(np.nanmean(nearest <= delta)),
        "mean_nearest_distance": float(np.mean(nearest)),
        "descriptor_clusters": connected_components(distances, delta),
        "coverage_radius": float(np.max(support_distance)),
        "archive_mean_Q": float(np.mean([item.quality for item in archive])),
        "archive_feasible_ratio": float(np.mean([item.feasible for item in archive])),
    }


def connected_components(distances: np.ndarray, delta: float) -> int:
    """Count connected components of the descriptor-neighborhood graph."""

    n = distances.shape[0]
    if n == 0:
        return 0
    adjacency = distances <= delta
    np.fill_diagonal(adjacency, True)
    visited = np.zeros(n, dtype=bool)
    components = 0
    for start in range(n):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(neighbor)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
    return components


def _split_parameters(split: str, index: int) -> dict[str, float | int]:
    """Return the deterministic collection conditions for one split."""

    if split == "demonstrations":
        return {
            "initial_xy_std": 0.055,
            "initial_yaw_std": 0.070,
            "observation_noise": 0.0030,
            "disturbance_std": 0.0015,
            "dynamics_scale": 1.0 + 0.010 * ((index % 5) - 2),
        }
    if split == "calibration":
        return {
            "initial_xy_std": 0.040,
            "initial_yaw_std": 0.050,
            "observation_noise": 0.0020,
            "disturbance_std": 0.0010,
            "dynamics_scale": 1.0,
        }
    if split == "candidate":
        return {
            "initial_xy_std": 0.050,
            "initial_yaw_std": 0.060,
            "observation_noise": 0.0020,
            "disturbance_std": 0.0012,
            "dynamics_scale": 1.0 + 0.015 * ((index % 7) - 3),
        }
    if split == "heldout":
        return {
            "initial_xy_std": 0.065,
            "initial_yaw_std": 0.075,
            "observation_noise": 0.0035,
            "disturbance_std": 0.0018,
            "dynamics_scale": 1.0 + 0.020 * ((index % 9) - 4),
        }
    raise ValueError(f"Unknown split: {split}")


def generate_split(
    split: str,
    count: int,
    seed: int,
    goal: np.ndarray,
) -> list[Trajectory]:
    """Generate one evaluator-data split without writing it to disk."""

    rng = np.random.default_rng(seed)
    trajectories = []
    mode_weights = np.array([0.28, 0.22, 0.15, 0.10, 0.08, 0.07, 0.07, 0.03])
    for index in range(count):
        mode_index = int(rng.choice(len(MODES), p=mode_weights))
        params = _split_parameters(split, index)
        trajectories.append(
            generate_trajectory(
                MODES[mode_index],
                goal,
                seed * 10000 + index,
                **params,
            )
        )
    return trajectories


def load_npz_protocol(path: Path) -> dict[str, list[Trajectory] | np.ndarray]:
    """Load state-only trajectories from an optional user-provided NPZ file."""

    result: dict[str, list[Trajectory] | np.ndarray] = {}
    with np.load(path, allow_pickle=True) as data:
        for split, expected in RECORDED_COUNTS.items():
            if split not in data:
                raise ValueError(
                    f"{path} must contain an array named '{split}'."
                )
            values = data[split]
            if len(values) != expected:
                raise ValueError(
                    f"{split}: expected {expected} trajectories, found {len(values)}."
                )
            trajectories = []
            for index, states in enumerate(values):
                states = np.asarray(states, dtype=float)
                if states.ndim != 2 or states.shape[1] < 3 or len(states) < 2:
                    raise ValueError(
                        f"{split}[{index}] must have shape (T, 3) with T >= 2."
                    )
                if not np.isfinite(states[:, :3]).all():
                    raise ValueError(f"{split}[{index}] contains non-finite values.")
                trajectories.append(
                    Trajectory(states=states[:, :3], source=split)
                )
            result[split] = trajectories
        target = np.asarray(data["target"] if "target" in data else GOAL, dtype=float)
    if target.shape != (2,):
        raise ValueError("target must be a two-element array.")
    result["target"] = target
    return result


def build_protocol(seed: int, input_npz: Path | None) -> dict:
    """Construct, calibrate, and freeze one independent protocol."""

    if input_npz is None:
        raw = {
            split: generate_split(split, count, seed + offset, GOAL)
            for offset, (split, count) in enumerate(SYNTHETIC_COUNTS.items())
        }
        target = GOAL.copy()
        split_counts = SYNTHETIC_COUNTS
    else:
        raw = load_npz_protocol(input_npz)
        target = np.asarray(raw["target"], dtype=float)
        split_counts = RECORDED_COUNTS

    encoder_seed = 260919 if input_npz is not None else seed
    encoder_n_init = 5 if input_npz is not None else 1
    encoder = StateVisitationEncoder(
        components=GMM_COMPONENTS,
        seed=encoder_seed,
        n_init=encoder_n_init,
    )
    encoder.fit(raw["demonstrations"] + raw["calibration"])
    encoded = {
        split: annotate(raw[split], encoder, target)
        for split in split_counts
    }
    calibration_descriptors = np.vstack(
        [item.descriptor for item in encoded["calibration"]]
    )
    delta = float(
        np.quantile(nearest_distances(calibration_descriptors), DELTA_QUANTILE)
    )
    return {
        "target": target,
        "encoder": encoder,
        "delta": delta,
        "split_counts": split_counts,
        **encoded,
    }


def rows_for_protocol(protocol: dict, seed: int) -> list[dict[str, object]]:
    candidates = protocol["candidate"]
    support = np.vstack([item.descriptor for item in candidates])
    rows = []
    for method_index, method in enumerate(METHODS):
        archive = select_archive(
            candidates,
            method,
            ARCHIVE_BUDGET,
            protocol["delta"],
            seed + 100 + method_index,
        )
        rows.append(
            {
                "seed": seed,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "delta": protocol["delta"],
                **archive_metrics(archive, support, protocol["delta"]),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        result: dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_trials": len(selected),
        }
        metrics = (
            "redundancy_ratio",
            "mean_nearest_distance",
            "descriptor_clusters",
            "coverage_radius",
            "archive_mean_Q",
            "archive_feasible_ratio",
        )
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            result[f"{metric}_mean"] = float(np.nanmean(values))
            valid = values[np.isfinite(values)]
            result[f"{metric}_std"] = (
                float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
            )
        summary.append(result)
    return summary


def write_metadata(
    path: Path,
    seeds: Sequence[int],
    input_npz: Path | None,
    deltas: dict[int, float],
    split_counts: dict[str, int],
) -> None:
    metadata = {
        "experiment": "minimal direct rule-based archive comparison",
        "data_mode": "user_npz" if input_npz else "deterministic_synthetic_surrogate",
        "input_npz": str(input_npz) if input_npz else None,
        "quality_evaluator": {
            "position_threshold_m": POSITION_THRESHOLD,
            "yaw_stability_threshold_rad": YAW_THRESHOLD,
            "reach_horizon_samples": REACH_HORIZON,
            "final_window_fraction": FINAL_WINDOW_FRACTION,
            "formula": "Q=min(1-e_s/0.15, 1-e_psi_s/0.15, 1-t_reach/500); q=I[Q>=0]",
        },
        "descriptor": {
            "features": "[x, y, sin(yaw), cos(yaw)]",
            "standardization": "fit on demonstrations plus calibration split",
            "model": "full-covariance GaussianMixture",
            "components": GMM_COMPONENTS,
            "fitting_seed": {
                "input_npz": 260919,
                "synthetic": "one protocol seed per run",
            },
            "n_init": {"input_npz": 5, "synthetic": 1},
            "delta_quantile": DELTA_QUANTILE,
        },
        "split_counts": split_counts,
        "archive_budget": ARCHIVE_BUDGET,
        "methods": list(METHODS),
        "random_seeds": [int(seed) for seed in seeds],
        "number_of_trials": {
            "independent_protocols": len(seeds),
            "archive_selections": len(seeds) * len(METHODS),
            "heldout_quality_records": len(seeds) * split_counts["heldout"],
            "generated_or_loaded_trajectory_records": (
                len(seeds) * sum(split_counts.values())
            ),
        },
        "delta_by_seed": {str(seed): value for seed, value in deltas.items()},
        "scope": "planar mechanism-level archive-composition demonstration",
        "paper_data_included": False,
        "physical_device_code_included": False,
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run(
    seeds: Sequence[int],
    output_dir: Path,
    input_npz: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    heldout_rows: list[dict[str, object]] = []
    deltas: dict[int, float] = {}
    split_counts: dict[str, int] | None = None

    for seed in seeds:
        print(f"seed={seed}", flush=True)
        protocol = build_protocol(seed, input_npz)
        current_counts = protocol["split_counts"]
        if split_counts is None:
            split_counts = current_counts
        elif split_counts != current_counts:
            raise RuntimeError("Split counts changed during one run.")
        deltas[int(seed)] = float(protocol["delta"])
        all_rows.extend(rows_for_protocol(protocol, seed))
        for index, item in enumerate(protocol["heldout"]):
            metrics = item.metrics or {}
            heldout_rows.append(
                {
                    "seed": seed,
                    "index": index,
                    "source": item.source,
                    "Q": item.quality,
                    "q": item.feasible,
                    **metrics,
                }
            )

    write_csv(output_dir / "fixed_budget_results.csv", all_rows)
    write_csv(output_dir / "fixed_budget_summary.csv", summarize(all_rows))
    write_csv(output_dir / "heldout_quality.csv", heldout_rows)
    assert split_counts is not None
    write_metadata(
        output_dir / "metadata.json",
        seeds,
        input_npz,
        deltas,
        split_counts,
    )
    print(f"results written to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the data-free public archive-comparison reproduction."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Independent protocol seeds; default: 11 22 33 44 55.",
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=None,
        help="Optional NPZ with demonstrations/calibration/candidate/heldout arrays.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.seeds, args.output_dir, args.input_npz)
