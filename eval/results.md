# Evaluation results

Hardware: CPU, 10 CPUs, Python 3.12.0

## Text (google/civil_comments (test), balanced sample, n=2000, ROC-AUC=0.919)

Threshold 0.05 was chosen for best F1 on a separate validation sample (n=2000); the test set was not used for tuning.

| Threshold | Accuracy | Precision | Recall | F1 | Macro-F1 |
|---|---|---|---|---|---|
| 0.5 | 70.2% | 95.1% | 42.7% | 58.9% | 67.8% |
| 0.05 | 83.9% | 85.8% | 81.3% | 83.5% | 83.9% |

## Latency per request (model inference, excludes network)

| Request | n | p50 | p95 |
|---|---|---|---|
| text single request | 100 | 57 ms | 61 ms |
| image single request | 50 | 233 ms | 283 ms |
| text plus image | 20 | 295 ms | 329 ms |
