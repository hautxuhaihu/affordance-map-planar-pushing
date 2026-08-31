# Recorded Experimental Environment

The dissertation experiments were run on Windows using PowerShell with the following environment:

| Component | Recorded value |
| --- | --- |
| Python | 3.10.19 |
| MuJoCo | 3.5.0 |
| NumPy | 2.2.5 |
| SciPy | 1.15.3 |
| PyTorch | 2.11.0+cu128 |
| CUDA reported by PyTorch | 12.8 |
| Matplotlib | 3.10.8 |
| pytest | 9.1.1 |
| Processor | Intel Core Ultra 7 255H |
| Graphics processor | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB |
| System memory | 32 GB |

`requirements.txt` records the Python package versions used by the public code. The reported workstation used the CUDA 12.8 build of PyTorch. A CPU build can run the focused tests and small checks, but the Joint and continuous full evaluations were designed for CUDA execution.

Check an installed environment with:

```powershell
python -X utf8 -c "import sys, mujoco, numpy, scipy, torch, matplotlib, pytest; print(sys.version); print('mujoco', mujoco.__version__); print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('torch', torch.__version__); print('cuda', torch.version.cuda); print('matplotlib', matplotlib.__version__); print('pytest', pytest.__version__)"
```

The formal worker counts and stage order are recorded in `REPRODUCE.md`.

