**Cost-Derived Affordance Closed-Loop Evaluation**

Dataset role: `test`. Each scenario uses paired episodes with the existing `belief_marginalised_closed_loop` controller.

| Scenario | Episodes | Success | Mean pushes | Mean final cost | Comparator cost | Cost improvement (95% CI) | AUC difference (95% CI) | Win / tie / loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Friction | 4096 | 100.00% | 2.2761 | 0.321217 | 0.321237 | 0.000021 [-0.000149, 0.000178] | 0.000000 [0.000000, 0.000000] | 9 / 4083 / 4 |
| Com | 4096 | 100.00% | 2.3311 | 0.328976 | 0.329109 | 0.000132 [-0.002496, 0.002540] | 0.000098 [-0.000159, 0.000427] | 104 / 3905 / 87 |
| Joint | 4096 | 100.00% | 2.5469 | 0.358644 | 0.358724 | 0.000080 [-0.000996, 0.001414] | -0.000012 [-0.000220, 0.000195] | 22 / 4049 / 25 |

Cost improvement is defined as comparator final cost minus cost-derived final cost. The confidence intervals use the fixed condition-by-target two-way paired bootstrap specified by the evaluation protocol.

The cost-derived affordance controller preserves closed-loop task completion while making the probability map and final action ordering the same cost-derived calculation. Confidence intervals containing zero are not interpreted as evidence of improved task performance.
