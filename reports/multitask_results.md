# GT-Bench Multitask Results

Selected checkpoint: `joint_adv_targeted_retention` (91.67% suite, 99.80% canonical, 99.60% robustness).

## Candidate Sweep

| Run | Canonical | Robustness | Suite |
| --- | ---: | ---: | ---: |
| `joint_base_canon_adv_suite` | 99.80% | 99.60% | 84.67% |
| `joint_adv_suite_retention` | 100.00% | 100.00% | 89.33% |
| `joint_adv_targeted_retention` | 99.80% | 99.60% | 91.67% |
| `joint_base_full_targeted` | 97.20% | 94.40% | 93.00% |
| `joint_adv_targeted_retention_seed1009` | 100.00% | n/a | 93.00% |
| `joint_adv_targeted_retention_seed2027` | 100.00% | n/a | 93.00% |

## Experiment Coverage

| Surface | Expected | Status | Notes |
| --- | --- | --- | --- |
| Candidate sweep | 4 recipes x 5 evaluations | complete | canonical, confirmation, stress, robustness, suite |
| Selected-recipe seeds | 3 seeds x 2 evaluations | complete | canonical and suite for seeds 42, 1009, 2027 |
| External base models | 2 models x 2 evaluations | complete | canonical and suite, base-only |
| Conditional second-round SFT | retention/suite-triggered only | not_required | selected checkpoint already clears canonical>=99.0%, robustness>=95.0%, and suite>=68.0% |
| Scope boundary | predefined matrix | documented | complete for the predefined recipe/evaluation matrix; not an exhaustive hyperparameter or all-game-theory benchmark search |

## Candidate Evaluation Matrix

| Run | Canonical | Confirmation | Stress | Robustness | Suite |
| --- | --- | --- | --- | --- | --- |
| `joint_base_canon_adv_suite` | yes | yes | yes | yes | yes |
| `joint_adv_suite_retention` | yes | yes | yes | yes | yes |
| `joint_adv_targeted_retention` | yes | yes | yes | yes | yes |
| `joint_base_full_targeted` | yes | yes | yes | yes | yes |

## Multi-Seed Summary

| Recipe | Runs | Suite mean | Suite seed SD | Canonical mean | Canonical seed SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `joint_adv_targeted_retention` | 3 | 92.56% | 0.63 pp | 99.93% | 0.09 pp |

| Recipe | Family | Suite seed SD |
| --- | --- | ---: |
| `joint_adv_targeted_retention` | `dominance` | 1.63 pp |
| `joint_adv_targeted_retention` | `extensive_form` | 0.00 pp |
| `joint_adv_targeted_retention` | `large_normal_form` | 3.40 pp |
| `joint_adv_targeted_retention` | `mixed_2x2` | 2.49 pp |
| `joint_adv_targeted_retention` | `natural_language` | 0.00 pp |
| `joint_adv_targeted_retention` | `repeated_interaction` | 0.00 pp |

## External Base Models

| Model | Canonical | Suite |
| --- | ---: | ---: |
| `Qwen/Qwen3-8B` | 2.60% | 4.00% |
| `Qwen/Qwen3-30B-A3B` | 3.80% | 6.33% |

## Selected Failure Diagnostics

Canonical failure counts: `tie_heavy_error`=1, `false_negative_profile`=1.
Suite failures by family: `dominance`=16, `large_normal_form`=7, `mixed_2x2`=2.
