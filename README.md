# Gravitomagnetic network identifiability

Python implementation of finite-network identifiability calculations for
directional time-transfer measurements in a stationary weak gravitational
field.

The code provides:

- straight-link and closed-loop gravitomagnetic response calculations;
- cycle-space design matrices;
- nuisance projection and rank tests;
- singular-value and Cramér--Rao stability diagnostics;
- source-current bounds and multipole-mismatch experiments;
- an Earth/Galileo-scale geometry benchmark.

## Requirements

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Quick check

```bash
python -m pytest -q
python src/run_experiments.py --mode smoke
python src/run_physical_campaign.py --mode smoke
python src/validate_smoke_results.py
```

Smoke mode checks the installation with reduced numerical settings. It is not
used for the reported numerical results.

## Full numerical run

The numerical results reported in the accompanying study can be regenerated with:

```bash
python src/run_experiments.py --mode full
python src/run_physical_campaign.py --mode full
python -m pytest -q
python src/validate_full_results.py
```
## Source layout

- `src/network_model.py` contains the numerical model and linear-algebra tools.
- `src/run_experiments.py` runs the dimensionless experiments.
- `src/run_physical_campaign.py` runs the physical-scale benchmark.
- `src/validate_smoke_results.py` and `src/validate_full_results.py` check the
  generated output.
- `tests/` contains the automated tests.

## Numerical convention

The core model uses geometrized units, `G = c = 1`, with

```text
A(x) = 2 integral J(x') / |x - x'| d^3x'
R_Gamma = -4 integral_Gamma A . dx
```

The physical benchmark restores SI factors explicitly.
## License

Released under the MIT License.

## Contact

Dino Vlahek — dvlahek@foi.hr
