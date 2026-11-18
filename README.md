**LPHAS: A Latency-Predictive Human–Agent Symbiosis Framework for Trust-Calibrated Autonomy in Deep-Space**
LPHAS is a research framework for human–agent collaboration in interplanetary missions, where communication delays can make real-time ground-in-the-loop control impractical. It combines latency- and hazard-aware autonomy allocation, an onboard predictive shadow model, and compact narrative telemetry intended to help ground operators understand and audit onboard decisions.
Research scope: This repository accompanies a framework-and-analysis study. The reported experiments use synthetic telemetry and simulation; they are not flight results or human-subject results. Resource figures are design targets, not hardware measurements.

Research overview
The framework addresses three connected questions:
1. How much authority should be delegated onboard? LPHAS computes a bounded, dimensionless Autonomy Coefficient (Ac) using round-trip communication latency, a hazard index, and an operator-conservatism parameter. The coefficient selects one of five autonomy levels, AL0–AL4.
2. For how long can onboard predictions be trusted? A shadow model is rolled forward to estimate its fidelity horizon. The study evaluates alternative error-growth models rather than assuming exponential degradation.
3. What should be sent to Earth? Narrative telemetry packages event information, actions, confidence, and a concise explanation in a compact format for delayed communications.
Framework architecture
LPHAS is organised into three tiers:
- Tier 1 — Perception and Data Gate: windows and normalises sensor streams, checks data validity, handles short gaps, and marks unreliable windows rather than silently passing invalid data downstream.
- Tier 2 — Shadow Logic: contains the anomaly-detection engine, the predictive shadow twin, and an event-triggered causal-analysis module. The policy layer uses hazard and prediction-fidelity information when selecting autonomy.
- Tier 3 — Narrative Telemetry: creates compact event-level packets for delayed downlink. The design uses structured records and template-first summaries rather than unrestricted free-form generation.
Ground control retains oversight, including the ability to review delegated actions, veto where applicable, revise the action envelope, and update the trust ledger at communication opportunities.
Autonomy levels
The policy defines five levels of authority:
Level	Authority allocation
AL0	Ground command only
AL1	Onboard system proposes; ground approves each action
AL2	Onboard system acts within the pre-approved envelope and reports
AL3	Onboard system acts within the envelope; ground may veto afterwards
AL4	Emergency authority beyond the envelope, subject to a hazard gate


The policy uses thresholds, hysteresis to reduce rapid switching, and an additional hazard condition for AL4. The paper treats these thresholds as design parameters that require mission-specific calibration.
Main mechanisms studied
- Delay-indexed delegation: a cost-based rule for allocating authority under communication delay.
- Rehearse-then-commit gating: candidate actions are checked against the shadow model before execution, with a deadline override.
- Counterfactual audit packets: downlink information intended to help operators understand why an action was selected and what alternatives were rejected.
- Evidence-based trust ledger: trust is updated from outcomes of delegated actions rather than being treated as a score to maximise.
Implementation and experimental setup
The accompanying research package is described as containing:
- A synthetic telemetry generator with fixed random seeds.
- A NumPy implementation of an LSTM forecaster.
- Nonparametric dynamic thresholding (NDT) and comparison detectors.
- Shadow-twin rollout experiments.
- Trust-ledger and mission-level co-simulation experiments.
- Generated figures and experiment outputs.
The manuscript reports Python 3.12, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.8.0, and Matplotlib 3.10.8 for the experimental environment. Check the package's dependency file and actual environment before installing; the versions above are the versions recorded in the manuscript.
Reproducibility
The study uses five independent seeds (0–4). Its synthetic telemetry testbed generates eight channels at one notional sample per minute, with nominal, storm, and injected-anomaly periods. The manuscript describes a 20,000-step training segment, a 6,000-step validation segment, and a 14,000-step test segment.

Once the code package is extracted, use the scripts and instructions included in that package to run each experiment. Exact script names and commands should be documented here after verifying the repository contents; they are intentionally not guessed in this README.
Selected findings
All figures below are from the synthetic testbed or analytical/simulation models described in the manuscript.
Experiment	Reported result	Interpretation
LSTM anomaly detection	Event F1 0.41 ± 0.02; persistence baseline 0.40 ± 0.02	The LSTM did not demonstrate a meaningful advantage over persistence on this testbed.
INT8 weight quantisation	0.41 event F1; 5,932 bytes of weights (26.9% of FP32 size)	Simulated INT8 quantisation preserved the reported detection score. This is not a measurement of flight-hardware latency or energy.
Shadow-twin fidelity	Power-law exponent β = 0.78 (calm) and 0.49 (storm)	The power-law model fit the reported rollout errors better than the linear-error alternative in the tested regimes.
Trust ledger	False trust reported as 0.9%, compared with 31% for posterior mean and 94% for the unqualified estimate described in the study	The reduction comes with slower trust accumulation.
Mission-level co-simulation	Mean event cost 42% lower than ground-only operation	This is a simulation result and depends on the cost-model assumptions.


Important negative and limiting results
The study also reports that:
- The forecaster was nearly indistinguishable from the persistence baseline.
- The residual-based detector missed many persistent faults, including level shifts, stuck sensors, and drifts. The paper therefore identifies non-residual hazard indicators as necessary.
- Rehearse-then-commit gating involves a safety–timeliness trade-off and requires a deadline override.
- Narrative telemetry's effect on operator situation awareness has not been tested with human participants.
- The resource budget, including the 10 W and 64 MB target, is a design allocation rather than a measured hardware result.
These limitations are part of the research findings and should be considered when interpreting the results.
Limitations and intended next steps
The manuscript identifies several open research directions:
- Improve coverage of residual-blind anomalies.
- Develop a coverage-guaranteed shadow-fidelity horizon.
- Formally verify autonomy switching and action-envelope rules.
- Evaluate narrative telemetry with a human-subject situation-awareness study.
- Validate resource, timing, and energy requirements on representative hardware.
- Replace simplified orbital assumptions with higher-fidelity ephemerides.
No human participants or flight telemetry were used in the reported work. Human-performance, trust, and situation-awareness claims remain hypotheses pending the proposed study and appropriate ethics approval.
Authors and affiliations
- Anusha Hasan — Karachi Institute of Engineering & Technology (KIET), Karachi, Pakistan
- Falak Shah — Karachi Institute of Engineering & Technology (KIET), Karachi, Pakistan
- Abdul Karim Hasan — Karachi Institute of Engineering & Technology (KIET), Karachi, Pakistan
- Rafia Kamran — Karachi Institute of Engineering & Technology (KIET), Karachi, Pakistan
- Arsalan Mehmood — Folio3 Software, Karachi, Pakistan
Correspondence: anushahasan02@gmail.com
Associated manuscript
Title: LPHAS: A Latency-Predictive Human–Agent Symbiosis Framework for Trust-Calibrated Autonomy in Deep-Space

Add the final publication or preprint URL, DOI, venue, and bibliographic details here when they are confirmed.
Citation
Please cite the associated manuscript once its publication details are final. A provisional BibTeX entry is provided below; update the entry to match the final publication record before using it in scholarly work.
@unpublished{hasan2026lphas,
  title  = {LPHAS: A Latency-Predictive Human-Agent Symbiosis Framework for Trust-Calibrated Autonomy in Deep-Space},
  author = {Hasan, Anusha and Shah, Falak and Hasan, Abdul Karim and Kamran, Rafia and Mehmood, Arsalan},
  year   = {2026},
  note   = {Research manuscript}
}
Data and code availability
The manuscript states that the data are synthetic and generated by the accompanying code package (lphas_code.zip), which includes the generator, NumPy LSTM, NDT, experiments, and figures. Add the public repository URL here after the code has been uploaded and checked.
License
No license is specified in the manuscript. Add a license only after the authors have agreed on the terms under which the code may be reused.
