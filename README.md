# LPHAS reference code (NumPy / SciPy / scikit-learn only)

All data are SYNTHETIC. Results show component behaviour under controlled conditions, not flight performance.

## Files
- `lphas_lib.py`  — telemetry generator, NumPy LSTM with hand-written BPTT + Adam, NDT thresholding, event-wise evaluation, weight quantisation.
- `exp_anomaly.py` — train/evaluate per seed:  `python exp_anomaly.py 0 1 2 3 4`  (about 1 min per seed on one CPU core; run seeds in separate calls)
- `exp_ledger.py` — trust-ledger Monte Carlo (S1-S3, forgetting factors).
- `exp_fidelity_mission.py` — rollout-fidelity fits (linear / power / saturating) and the mission-level co-simulation (needs `results_anomaly.json`).
- `figs.py` — diagrams and result figures (needs `ex0.npz` from seed 0).
- `analytic_policy_sim.py`, `delegation_sweep.py` — analytical A_c / MTAR figures and the delegation-cost sweep.
- `results_*.json` — the exact outputs reported in the paper.

## Order
`python exp_anomaly.py 0 1 2 3 4` -> `python exp_ledger.py` -> `python exp_fidelity_mission.py` -> `python figs.py`

Create a `fig/` folder next to the scripts before running (one is included). Seeds are fixed; re-running reproduces the JSON files up to library-version differences.
Software used: Python 3.12, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.8.0, Matplotlib 3.10.8.
