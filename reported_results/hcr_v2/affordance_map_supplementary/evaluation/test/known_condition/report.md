# Cost-Derived Affordance Map: Known Condition

- Role: `test`
- Protocol: `cost_derived_affordance_map_v1`
- Temperature: `0.1`

## Scenario Summary

| Scenario | Decisions | Top-1 mean regret | Top-20 exact coverage | Top-100 exact coverage | Top-100 near-0.05 | Region/action mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| friction | 2048 | 0.024761 | 99.90% | 100.00% | 100.00% | 0.00% |
| com | 2048 | 0.053678 | 100.00% | 100.00% | 100.00% | 3.08% |
| joint | 2048 | 0.047140 | 99.85% | 100.00% | 100.00% | 3.27% |

## Interpretation Boundary

The probabilities are a monotonic representation of predicted posterior-expected TNPO cost. They are not calibrated physical success probabilities and do not establish a global optimum outside the 4,536-action library.
