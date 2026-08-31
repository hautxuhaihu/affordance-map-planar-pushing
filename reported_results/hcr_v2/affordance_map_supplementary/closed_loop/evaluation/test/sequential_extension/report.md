**Cost-Derived Affordance Closed-Loop Evaluation**

Dataset role: `test`. Each scenario uses paired episodes with the existing `belief_marginalised_closed_loop` controller.

| Scenario | Episodes | Success | Mean pushes | Mean final cost | Comparator cost | Cost improvement (95% CI) | AUC difference (95% CI) | Win / tie / loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Friction | 2048 | 100.00% | 2.8101 | 0.321716 | 0.321722 | 0.000006 [-0.000334, 0.000293] | 0.000000 [0.000000, 0.000000] | 7 / 2037 / 4 |
| Com | 2048 | 100.00% | 3.0562 | 0.327245 | 0.327440 | 0.000195 [-0.005175, 0.004907] | 0.000195 [-0.000342, 0.000879] | 91 / 1877 / 80 |
| Joint | 2048 | 100.00% | 3.2202 | 0.343623 | 0.343784 | 0.000161 [-0.001979, 0.002717] | -0.000024 [-0.000415, 0.000391] | 22 / 2001 / 25 |

Cost improvement is defined as comparator final cost minus cost-derived final cost. The confidence intervals use the fixed condition-by-target two-way paired bootstrap specified by the evaluation protocol.

The cost-derived affordance controller preserves closed-loop task completion while making the probability map and final action ordering the same cost-derived calculation. Confidence intervals containing zero are not interpreted as evidence of improved task performance.
