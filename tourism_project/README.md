# Visit with Us - Wellness Tourism Package: MLOps pipeline

Predicts whether a customer will purchase the newly introduced **Wellness Tourism Package** before the sales team contacts them.

## Repository structure
```
.github/workflows/pipeline.yml      GitHub Actions workflow (CI/CD)
tourism_project/
  data/tourism.csv                  raw dataset
  model_building/
    data_register.py                validates the dataset schema and prints a summary
    prep.py                         cleans the data, creates train/test splits
    train.py                        tunes XGBoost, logs to MLflow, saves the best model
  deployment/
    app.py                          Streamlit app (loads the model committed by the pipeline)
    requirements.txt                dependencies of the Streamlit app
    best_tourism_model_v1.joblib    trained model (committed by the pipeline)
  requirements.txt                  dependencies of the pipeline jobs
```

## Pipeline
`register-dataset` -> `data-prep` -> `model-training`. The splits are passed between jobs as an artifact and the
trained model is committed back to `main`. The app is deployed on Streamlit Community Cloud from
`tourism_project/deployment/app.py` (Python 3.11).
