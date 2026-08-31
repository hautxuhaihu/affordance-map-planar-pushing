# Cost-Derived Affordance Map: Closed Loop

- Role: `test`
- Protocol: `cost_derived_affordance_map_v1`
- Temperature: `0.1`

## Scenario Summary

| Scenario | Decisions | Region/action mismatch | Mean map time (s) |
| --- | ---: | ---: | ---: |
| friction | 975 | 9.23% | 0.000827 |
| com | 2420 | 9.46% | 0.000754 |
| joint | 3376 | 9.09% | 0.009983 |

## Interpretation Boundary

The probabilities are a monotonic representation of predicted posterior-expected TNPO cost. They are not calibrated physical success probabilities and do not establish a global optimum outside the 4,536-action library.
