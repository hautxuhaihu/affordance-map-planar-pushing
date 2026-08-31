# Reported Dissertation Results

This directory contains the principal saved summaries cited by Appendix A of the dissertation.

The files were copied from the fixed local result locations without recalculating or changing their values. They provide traceability for the main tables and claims, but they do not replace the larger decision-level records, model checkpoints or MuJoCo datasets used to generate them.

The directory preserves the original relative structure beneath `results/`. For example:

```text
reported_results/hcr_v2/e1/evaluation/test/combined_summary.json
reported_results/car/experiment_4_v2/evaluation/test/combined_summary.json
```

Use `SHA256SUMS` to verify the files after download.

| Dissertation evidence | Saved summary |
| --- | --- |
| Discrete object-motion prediction and ranked-first action quality | `hcr_v2/e1/evaluation/test/combined_summary.json` |
| Bayesian estimation of friction and COM | `hcr_v2/e2/test/combined_summary.json` |
| Affordance map with supplied friction and COM | `hcr_v2/affordance_map_supplementary/evaluation/test/known_condition/report.md` |
| Affordance map using the Bayesian belief | `hcr_v2/affordance_map_supplementary/evaluation/test/belief_conditioned/report.md` |
| Affordance-map sequence during repeated pushing | `hcr_v2/affordance_map_supplementary/evaluation/test/closed_loop/report.md` |
| Repeated-pushing evaluation on all target poses | `hcr_v2/affordance_map_supplementary/closed_loop/evaluation/test/all/report.md` |
| Multi-push target-pose evaluation | `hcr_v2/affordance_map_supplementary/closed_loop/evaluation/test/sequential_extension/report.md` |
| Available improvement from continuous refinement | `car/experiment_1/continuous_action_refinement_summary.json` |
| Continuous object-motion prediction | `car/experiment_2/evaluation/test/summary.json` |
| Continuous candidate-budget selection | `car/experiment_3/evaluation/selected_configuration.json` |
| Condition-dependent continuous predictor selection | `car/experiment_4_v2/models/backend_selection_summary.json` |
| Continuous likelihood calibration | `car/experiment_4_v2/continuous_likelihood/selected_configuration.json` |
| Repeated pushing with continuous refinement | `car/experiment_4_v2/evaluation/test/combined_summary.json` |
