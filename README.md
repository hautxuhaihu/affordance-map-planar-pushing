# Physics-based Affordance Map for Planar Pushing Motion

This repository contains the code associated with the MSc dissertation **Physics-based Affordance Map for Planar Pushing Motion**.

The project studies how a planar pushing action can be selected for a target pose when object-table friction and planar centre-of-mass (COM) offset may initially be unknown. A structured 4,536-action library is evaluated through predicted task cost. The costs are converted into a probability-based affordance map, while observations from earlier pushes update a Bayesian belief over friction and COM. The final experiments increase action resolution through bounded continuous refinement around promising discrete actions.

## Research scope

The repository supports the dissertation experiments for:

1. object-motion prediction with supplied friction and COM;
2. Bayesian estimation of unknown friction and planar COM offset;
3. probability-based affordance maps and Top-100 candidate selection;
4. repeated pushing using the Bayesian belief;
5. bounded continuous refinement of selected discrete actions.

The evidence is limited to the included MuJoCo simulation environment. It does not establish real-robot performance or a global optimum over unrestricted continuous actions.

## Repository structure

```text
affordance-map-planar-pushing/
├── assets/xml/          MuJoCo models and friction/COM variants
├── configs/             Small JSON configuration files
├── experiments/
│   ├── hcr_v2/          Discrete affordance, Bayesian estimation and repeated pushing
│   └── car/             Continuous action refinement experiments
├── manifests/
│   ├── hcr_v2/          Action, friction/COM and target-pose manifests
│   └── car/             Continuous-refinement action and target splits
├── src/push_core/       Shared simulation, prediction and evaluation code
└── tests/hcr_v2/        Focused semantic tests for the HCR V2 implementation
```

Large datasets, trained checkpoints and generated result files are not included in this repository.

## Tested environment

- Python 3.10.19
- MuJoCo 3.5.0
- NumPy 2.2.5
- SciPy 1.15.3
- PyTorch 2.11.0
- Matplotlib 3.10.8

The exact Python dependencies used for the release are listed in `requirements.txt`.

## Installation

Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/hautxuhaihu/affordance-map-planar-pushing.git
cd affordance-map-planar-pushing

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The experiment entry points add the local `src` directory to the Python import path automatically.

## Quick verification

Check the fixed action, condition and target-pose manifests without running MuJoCo rollouts:

```powershell
python -X utf8 experiments\hcr_v2\run_e1.py plan
```

Run the focused tests in isolated processes:

```powershell
python -X utf8 -m pytest -q tests\hcr_v2\test_e1.py
python -X utf8 -m pytest -q tests\hcr_v2\test_e2.py
python -X utf8 -m pytest -q tests\hcr_v2\test_e5.py
```

## Main experiment entry points

### Discrete affordance and Bayesian estimation

```text
experiments/hcr_v2/run_e1.py   Object-motion prediction with supplied friction and COM
experiments/hcr_v2/run_e2.py   Sequential friction and COM estimation
experiments/hcr_v2/run_e3.py   Top-100 candidate selection using the Bayesian belief
experiments/hcr_v2/run_e4.py   Ranked-first action evaluation
experiments/hcr_v2/run_e5.py   Repeated discrete pushing
```

The cost-derived affordance-map analysis and repeated-pushing controller are provided in:

```text
experiments/hcr_v2/analyse_affordance_maps.py
experiments/hcr_v2/plot_affordance_maps.py
experiments/hcr_v2/run_cost_derived_affordance_closed_loop.py
```

### Continuous action refinement

```text
experiments/car/run_e1.py      Available improvement near discrete actions
experiments/car/run_e2.py      Continuous object-motion prediction
experiments/car/run_e3.py      Nominal-condition continuous action selection
experiments/car/run_e4_v2.py   Continuous refinement with unknown friction and COM
```

Run an entry point with `--help` to view its available commands and parameters. The complete formal experiments generate large intermediate datasets and can require substantial CPU, GPU and storage resources.

## Data and generated results

Experiment outputs are written under:

```text
data/
results/
```

Both directories are ignored by Git. The included manifests define the action library, friction and COM settings, and target-pose splits required by the dissertation protocol.

## Citation

Citation details will be added after the final dissertation submission.

## License

This project is released under the MIT License. See `LICENSE` for details.
