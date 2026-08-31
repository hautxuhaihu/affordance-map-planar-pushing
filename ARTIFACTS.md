# Reproducibility Artefacts

## Included in Git

The repository contains:

- source code for the discrete and continuous experiments;
- MuJoCo XML models and JSON configuration files;
- fixed action, friction/COM and target-position manifests;
- focused semantic tests; and
- the principal dissertation result summaries under `reported_results/`.

The 13 reported-result files occupy approximately 0.4 MiB. Their checksums are recorded in `reported_results/SHA256SUMS`.

## Compact Release Bundle

The `v1.0-dissertation` GitHub release provides `affordance-map-planar-pushing-reproducibility-v1.0.zip` and its `.sha256` checksum file. The compact bundle contains:

- per-action tensor-interpolation artefacts;
- MLP comparator checkpoints and normalisers;
- Bayesian likelihood residual statistics;
- the internal proposal-model artefacts required by the shared repeated-pushing engine;
- continuous object-motion model checkpoints;
- condition-dependent continuous model checkpoints;
- selected Validation configurations; and
- continuous likelihood statistics.

The paths inside the archive begin with `results/`. Extract the archive at the repository root.

## Artefacts Not Included in Git

Large generated data and complete decision-level results remain outside Git:

| Artefact group | Approximate local size |
| --- | ---: |
| HCR V2 generated data | 1.63 GiB |
| HCR V2 complete results | 0.88 GiB |
| Continuous-refinement generated data | 2.94 GiB |
| Continuous-refinement complete results | 0.01 GiB |

These files can be regenerated using `REPRODUCE.md`. They should be placed in an external research-data archive if long-term access to every rollout and decision record is required.

## Traceability

The principal result paths are listed in `reported_results/README.md` and preserve the structure used in Appendix A. The compact release bundle is intended to reduce model reconstruction time. It does not claim to replace the full raw-data archive.
