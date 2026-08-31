# Reproducing the Dissertation Experiments

This document gives the dependency order for the experiments reported in the MSc dissertation *Physics-based Affordance Map for Planar Pushing Motion*. Commands are run from the repository root in Windows PowerShell.

## Reproduction Levels

Two levels of reproduction are supported:

1. **Result inspection:** use the small summaries under `reported_results/` to trace the principal dissertation tables and claims to saved outputs.
2. **Full regeneration:** collect the MuJoCo data, construct or train the models, select settings on Validation and evaluate the fixed settings on Test.

The full route is computationally expensive. The E1 discrete collection alone contains 1,356,264 MuJoCo rollouts. The original local artefacts occupied approximately 4.57 GiB under `data/` and 0.89 GiB under `results/`.

## Environment

Create the environment described in the main README:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that the fixed manifests can be read:

```powershell
python -X utf8 experiments\hcr_v2\run_e1.py plan
```

Run the semantic tests in separate processes:

```powershell
python -X utf8 -m pytest -q tests\hcr_v2\test_e1.py
python -X utf8 -m pytest -q tests\hcr_v2\test_e2.py
python -X utf8 -m pytest -q tests\hcr_v2\test_e5.py
```

The tests are intentionally separated because some Windows NumPy/MKL configurations can terminate when all three files run in one Python process.

## Optional Compact Artefact Bundle

The GitHub release includes a compact reproducibility bundle containing trained models, tensor-interpolation artefacts, normalisers, likelihood statistics and selected Validation configurations. Extract the bundle at the repository root so that its `results/` directory merges with the local `results/` directory.

The bundle avoids rebuilding the small model artefacts, but it does not replace the large MuJoCo datasets or recorded episode histories needed to rerun every evaluation.

## Stage 1: Discrete Object-Motion Prediction

Collect Training, Validation and Test motion data:

```powershell
python -X utf8 experiments\hcr_v2\run_e1.py collect --scenario all --role training --num-workers 8 --resume
python -X utf8 experiments\hcr_v2\run_e1.py collect --scenario all --role validation --num-workers 8 --resume
python -X utf8 experiments\hcr_v2\run_e1.py collect --scenario all --role test --num-workers 8 --resume
```

Construct the tensor-interpolation models and train the MLP comparator:

```powershell
python -X utf8 experiments\hcr_v2\run_e1.py prepare-p1 --scenario all
python -X utf8 experiments\hcr_v2\run_e1.py train-p2 --scenario all --num-workers 8
```

Evaluate Validation before Test:

```powershell
python -X utf8 experiments\hcr_v2\run_e1.py evaluate --scenario all --role validation --predictors p0,p1,p2 --bootstrap-resamples 10000
python -X utf8 experiments\hcr_v2\run_e1.py evaluate --scenario all --role test --predictors p0,p1,p2 --bootstrap-resamples 10000
```

## Stage 2: Bayesian Estimation of Friction and COM

Collect the likelihood-statistics data and fit prediction residuals:

```powershell
python -X utf8 experiments\hcr_v2\collect_e2_data.py training-outcomes --scenario all --num-workers 8 --resume
python -X utf8 experiments\hcr_v2\run_e2.py fit-residuals --scenario all
```

Collect and evaluate Validation histories before using the fixed settings on Test:

```powershell
python -X utf8 experiments\hcr_v2\collect_e2_data.py validation-histories --scenario all --num-workers 8 --resume
python -X utf8 experiments\hcr_v2\run_e2.py calibrate-student-t-validation --scenario all
python -X utf8 experiments\hcr_v2\run_e2.py compare-final-quadrature --scenario all
python -X utf8 experiments\hcr_v2\run_e2.py evaluate-validation --scenario all

python -X utf8 experiments\hcr_v2\collect_e2_data.py test-histories --scenario all --num-workers 8 --resume
python -X utf8 experiments\hcr_v2\run_e2.py evaluate-test --scenario all --bootstrap-resamples 10000
```

The fixed target-position manifests required by these commands are already included under `manifests/hcr_v2/`.

## Stage 3: Probability-Based Affordance Map

Evaluate the cost-derived affordance map with supplied friction and COM values and with the Bayesian belief:

```powershell
python -X utf8 experiments\hcr_v2\analyse_affordance_maps.py --analysis-mode known_condition --role test --scenario all --max-episodes 0
python -X utf8 experiments\hcr_v2\analyse_affordance_maps.py --analysis-mode belief_conditioned --role test --scenario all --max-episodes 0
python -X utf8 experiments\hcr_v2\analyse_affordance_maps.py --analysis-mode closed_loop --role test --scenario all --target-group sequential_extension --max-episodes 0 --require-update-four
```

Generate representative saved affordance-map figures after the analysis records exist:

```powershell
python -X utf8 experiments\hcr_v2\plot_affordance_maps.py --analysis-mode known_condition --role test --max-figures 1
python -X utf8 experiments\hcr_v2\plot_affordance_maps.py --analysis-mode closed_loop --role test --max-figures 1
```

Increase `--max-figures` when more saved episodes are required.

## Stage 4: Repeated Pushing

Validation must be completed before Test. Friction and COM can use two workers on the recorded workstation; Joint uses one worker because it represents more friction and COM values.

```powershell
python -X utf8 experiments\hcr_v2\run_cost_derived_affordance_closed_loop.py collect --role validation --scenario friction --target-group all --worker-mode process --num-workers 2 --resume
python -X utf8 experiments\hcr_v2\run_cost_derived_affordance_closed_loop.py collect --role validation --scenario com --target-group all --worker-mode process --num-workers 2 --resume
python -X utf8 experiments\hcr_v2\run_cost_derived_affordance_closed_loop.py collect --role validation --scenario joint --target-group all --worker-mode process --num-workers 1 --resume
python -X utf8 experiments\hcr_v2\run_cost_derived_affordance_closed_loop.py evaluate --role validation --scenario all --target-group all --bootstrap-resamples 10000

python -X utf8 experiments\hcr_v2\run_cost_derived_affordance_closed_loop.py collect --role test --scenario friction --target-group all --worker-mode process --num-workers 2 --resume
python -X utf8 experiments\hcr_v2\run_cost_derived_affordance_closed_loop.py collect --role test --scenario com --target-group all --worker-mode process --num-workers 2 --resume
python -X utf8 experiments\hcr_v2\run_cost_derived_affordance_closed_loop.py collect --role test --scenario joint --target-group all --worker-mode process --num-workers 1 --resume
python -X utf8 experiments\hcr_v2\run_cost_derived_affordance_closed_loop.py evaluate --role test --scenario all --target-group all --bootstrap-resamples 10000
```

## Stage 5: Continuous Action Refinement

### Experiment 1: Available Improvement

```powershell
python -X utf8 experiments\car\run_e1.py plan
python -X utf8 experiments\car\run_e1.py prepare
python -X utf8 experiments\car\run_e1.py collect --num-workers 8 --resume
python -X utf8 experiments\car\run_e1.py evaluate --bootstrap-resamples 10000
```

### Experiment 2: Continuous Object-Motion Prediction

```powershell
python -X utf8 experiments\car\run_e2.py prepare
python -X utf8 experiments\car\run_e2.py collect --num-workers 8 --resume
python -X utf8 experiments\car\run_e2.py train
python -X utf8 experiments\car\run_e2.py evaluate-validation --bootstrap-resamples 10000
python -X utf8 experiments\car\run_e2.py evaluate-test --bootstrap-resamples 10000
```

### Experiment 3: Continuous Selection with Supplied Friction and COM

```powershell
python -X utf8 experiments\car\run_e3.py prepare
python -X utf8 experiments\car\run_e3.py collect-validation --num-workers 8 --resume
python -X utf8 experiments\car\run_e3.py evaluate-validation --bootstrap-resamples 10000
python -X utf8 experiments\car\run_e3.py prepare-test
python -X utf8 experiments\car\run_e3.py collect-test --num-workers 8 --resume
python -X utf8 experiments\car\run_e3.py evaluate-test --bootstrap-resamples 10000
```

### Experiment 4: Continuous Refinement Using the Bayesian Belief

```powershell
python -X utf8 experiments\car\run_e4_v2.py prepare
python -X utf8 experiments\car\run_e4_v2.py collect-model-data --scenario all --role training --num-workers 8 --resume
python -X utf8 experiments\car\run_e4_v2.py collect-model-data --scenario all --role validation --num-workers 8 --resume
python -X utf8 experiments\car\run_e4_v2.py train-backends
python -X utf8 experiments\car\run_e4_v2.py evaluate-backends --bootstrap-resamples 10000
python -X utf8 experiments\car\run_e4_v2.py fit-likelihood
python -X utf8 experiments\car\run_e4_v2.py evaluate-likelihood --scenario all

python -X utf8 experiments\car\run_e4_v2.py collect --scenario friction --role validation --num-workers 2 --resume
python -X utf8 experiments\car\run_e4_v2.py collect --scenario com --role validation --num-workers 2 --resume
python -X utf8 experiments\car\run_e4_v2.py collect --scenario joint --role validation --num-workers 1 --resume
python -X utf8 experiments\car\run_e4_v2.py calibrate-on-policy-likelihood --scenario all
python -X utf8 experiments\car\run_e4_v2.py evaluate-exact-selector --scenario all
python -X utf8 experiments\car\run_e4_v2.py evaluate --scenario all --role validation --bootstrap-resamples 10000

python -X utf8 experiments\car\run_e4_v2.py collect --scenario friction --role test --num-workers 2 --resume
python -X utf8 experiments\car\run_e4_v2.py collect --scenario com --role test --num-workers 2 --resume
python -X utf8 experiments\car\run_e4_v2.py collect --scenario joint --role test --num-workers 1 --resume
python -X utf8 experiments\car\run_e4_v2.py evaluate --scenario all --role test --bootstrap-resamples 10000
```

`benchmark-workers` is available for hardware-specific throughput checks. It is not required when reproducing the recorded two-worker Friction/COM and one-worker Joint configuration.

## Validity Boundary

Validation selects models, likelihood settings, candidate budgets and promotion thresholds. Test evaluates those fixed choices. Changing manifests, seeds, physical settings, success tolerances, update limits or selected Validation configurations creates a new experiment rather than a reproduction of the reported results.
