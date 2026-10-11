# Minimal public code for rule-based archive comparison

This directory contains a small, data-free implementation of the
reproducibility-relevant archive protocol described in the manuscript:

- direct task-quality evaluation using the binary feasibility value `q` and
  continuous quality value `Q`;
- a fixed 12-component Gaussian-mixture state-visitation descriptor over
  `[x, y, sin(yaw), cos(yaw)]`;
- calibration of the redundancy threshold from a separate calibration split;
- fixed-budget comparison of `Uniform`, `Quality-only`, `Novelty-only`, and
  `Q + q + d` archive selection.

The release intentionally does **not** contain the manuscript's recorded
demonstrations, rollout files, physical-device interface, trained checkpoint,
or result files. By default, the script creates a deterministic synthetic
planar surrogate so that the complete evaluator and selection logic can be
run from a clean checkout. This synthetic run is a runnable protocol
demonstration; it is not intended to regenerate the numerical values in the
manuscript tables.

## Requirements

Python 3.11 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
```

## Run

From this directory:

```powershell
python .\run_public_reproduction.py
```

The default run uses the same five seed values reported for the controlled
simulation:

```text
11, 22, 33, 44, 55
```

It creates `results/` with:

- `fixed_budget_results.csv`: one row per seed and archive rule;
- `fixed_budget_summary.csv`: mean and standard deviation over seeds;
- `heldout_quality.csv`: direct evaluator output for the held-out split;
- `metadata.json`: locked settings, split counts, seeds, and trial counts.

To use different independent seeds:

```powershell
python .\run_public_reproduction.py --seeds 11 22 33 44 55
```

The public script locks the central fixed-budget comparison at `M=8`. It
implements the archive-composition part of the paper's main protocol:
evaluate each trajectory, fit and freeze the descriptor, calibrate `delta`,
and compare the archive rules on one common candidate pool. It deliberately
does not include the conditional-policy training, 12-condition validation
gate, three refinement rounds, physical deployment interface, or auxiliary
audits. Those components are outside this minimal release.

## Quality-evaluator data procedure

The code uses state sequences with columns `[x, y, yaw]`. In the default
data-free mode, four deterministic splits are generated:

| Split | Number of trajectories | Role |
|---|---:|---|
| demonstrations | 32 | initialization trajectories in the controlled simulation |
| calibration | 48 | GMM fitting and redundancy-threshold calibration |
| candidate | 240 | common fixed-budget archive pool |
| heldout | 26 | independent direct-quality check in this minimal release |

Each synthetic trajectory is generated for 200 steps from a planar
three-degree-of-freedom dynamical surrogate. The collection procedure varies
the initial position/yaw, observation noise, process disturbance, dynamics
scale, and one of eight trajectory-shape modes. The split-specific settings
are explicit in `_split_parameters`; the random seed for each trajectory is
deterministic. No generated trajectory is stored in the repository.

The manuscript's recorded-data path used 17 valid demonstrations rather than
the 32 synthetic initialization trajectories used by the controlled
simulation. To keep the private recorded data out of this release, the
optional NPZ interface below accepts the recorded-data counts
17/48/240/26. In that mode, the fixed descriptor uses analysis seed `260919`
and five GMM initializations, matching the recorded-data implementation.

For each trajectory, the evaluator uses the final 20% stabilization window:

```text
e_s       = mean final-window position error
e_psi_s   = circular-RMS final-window yaw fluctuation
t_reach   = first sample with position error <= 0.15 m
Q         = min(1-e_s/0.15, 1-e_psi_s/0.15, 1-t_reach/500)
q         = 1 when all three limits are satisfied, otherwise 0
```

The evaluator is deterministic for fixed states and target. It does not train
a quality classifier and does not use a learned proxy score.

## Relationship to the manuscript data

The manuscript's recorded quality-evaluator collection used 20 teleoperated
trials, of which 17 valid demonstrations were retained after quality
inspection. Policy-generated rollout records were then separated into 48
calibration records, 240 candidate records, and 26 held-out records. Files
were ordered lexicographically before the split, and the recorded-data
analysis used seed `260919` for the fixed descriptor and threshold
calibration. The same direct evaluator was applied to demonstrations and
rollouts.

Those recorded files are not included in this release. The default synthetic
generator is supplied only to make the code runnable without private data; it
uses the same split roles, evaluator, descriptor construction, and archive
selection interface. Therefore, this repository is a minimal implementation
of the reported protocol, not a release of the complete experimental dataset
or a claim that every manuscript table is regenerated from synthetic data.

## Optional user-provided state sequences

The same evaluator and archive rules can be applied to a local NPZ file.
This public directory does not include such a file. Create an NPZ with:

```python
import numpy

numpy.savez(
    "protocol.npz",
    demonstrations=numpy.asarray(demonstrations, dtype=object),
    calibration=numpy.asarray(calibration, dtype=object),
    candidate=numpy.asarray(candidate, dtype=object),
    heldout=numpy.asarray(heldout, dtype=object),
    target=numpy.asarray([x_goal, y_goal], dtype=float),
)
```

Each item in the four object arrays must be a floating-point array with shape
`(T, 3)` and columns `[x, y, yaw]`. The required counts are 17, 48, 240, and
26. Run:

```powershell
python .\run_public_reproduction.py --input-npz .\protocol.npz
```

To run one recorded-data-style analysis, use the fixed analysis seed:

```powershell
python .\run_public_reproduction.py --input-npz .\protocol.npz --seeds 260919
```

## Scope

This is a minimal reproducibility implementation of the archive-composition
mechanism. It is not a release of the private experiment data and it is not a
full implementation of the physical BlueROV2 deployment or every auxiliary
audit. It also does not include the conditional-policy training and
three-round validation loop used for the separate iterative simulation. The
synthetic run demonstrates the direct evaluator structure, descriptor
construction, random-seed handling, trial counts, and archive-selection rules
needed to inspect the paper's archive-composition mechanism.
