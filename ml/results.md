### Lenient match (text fields match if one contains the other)
| System | name | budgetSanctioned | budgetSpent | completionPercentage | leaderName | scheme | department | MICRO_P | MICRO_R | MICRO_F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Rule-based | 0.967 | 0.774 | 0.869 | 0.890 | 1.000 | 1.000 | 0.799 | 0.943 | 0.883 | 0.912 |
| BERT (fine-tuned) | 1.000 | 0.724 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 | 0.933 | 0.963 |
| Hybrid (BERT + rules) | 1.000 | 0.829 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 | 0.957 | 0.975 |

### Strict match (exact after normalisation)
| System | name | budgetSanctioned | budgetSpent | completionPercentage | leaderName | scheme | department | MICRO_P | MICRO_R | MICRO_F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Rule-based | 0.967 | 0.774 | 0.869 | 0.890 | 1.000 | 1.000 | 0.751 | 0.940 | 0.880 | 0.909 |
| BERT (fine-tuned) | 0.802 | 0.724 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.946 | 0.888 | 0.916 |
| Hybrid (BERT + rules) | 0.802 | 0.829 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.911 | 0.929 |
