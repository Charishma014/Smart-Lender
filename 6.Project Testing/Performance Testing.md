# Performance Testing

## Machine Learning Model
- **Accuracy:** The XGBoost model achieved ~85%+ accuracy on the test set.
- **Validation:** Cross-validation was used to ensure the model does not overfit.

## Web Application
- **Latency:** The prediction endpoint `/api/predict` processes requests and returns results in < 500ms.
- **Concurrency:** Flask development server tested with multiple simultaneous requests; performs adequately for a prototype.
- **Database:** MongoDB reads/writes are virtually instantaneous for the current data volume.