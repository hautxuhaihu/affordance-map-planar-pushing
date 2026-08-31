# Cost-Derived Affordance Map: Belief Conditioned

- Role: `test`
- Protocol: `cost_derived_affordance_map_v1`
- Temperature: `0.1`

## Scenario Summary

| Scenario | Affordance method | Decisions | Top-1 mean regret | Top-20 exact coverage | Top-100 exact coverage | Top-100 near-0.05 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| friction | nominal_condition_cost | 6734 | 0.910432 | 47.02% | 86.17% | 95.69% |
| friction | posterior_expected_cost | 6734 | 0.027421 | 95.65% | 98.74% | 99.91% |
| com | nominal_condition_cost | 6185 | 0.124227 | 78.17% | 97.57% | 98.90% |
| com | posterior_expected_cost | 6185 | 0.040013 | 97.48% | 99.94% | 99.97% |
| joint | nominal_condition_cost | 7099 | 0.803372 | 31.12% | 70.31% | 79.63% |
| joint | posterior_expected_cost | 7099 | 0.046701 | 95.51% | 99.59% | 99.92% |

## Interpretation Boundary

The probabilities are a monotonic representation of predicted posterior-expected TNPO cost. They are not calibrated physical success probabilities and do not establish a global optimum outside the 4,536-action library.
