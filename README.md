# Telco Churn Prediction - Modeling Pipeline

## Overview
This project implements a complete machine learning pipeline to predict customer churn in a telco compay context. 
The pipeline follows best practices for reproducibility, interpretability and performance optimization.

## Project Structure
```
data/
    raw/           # Original datasets
    processed/     # Preprocessed datasets (train, eval, test splits)
models/
    final_model.joblib
notebooks/
    01_data_exploration.ipynb
    02_preprocessing_modeling.ipynb
reports/
    model_selection_results.csv
    test_metrics.json
    figures/
        confusion_matrix_test.png
        pr_curve_test.png
        roc_curve_test.png
src/
    data_preprocessing.py
```

## Workflow
1. **Data Preprocessing**  
   - Cleaning, encoding categorical variables, scaling numeric features.
   - Saving preprocessed splits (`X_train`, `X_eval`, `X_test`, and corresponding `y` files).

2. **Model Selection**  
   - Hyperparameter tuning via `RandomizedSearchCV` for:
     - Logistic Regression
     - Random Forest Classifier
     - XGBoost Classifier
   - Evaluation metrics: F1-score, Precision, Recall, Accuracy, ROC-AUC.

3. **Final Training**  
   - Best model retrained on combined `train + eval` data.
   - Final metrics computed on test set.

4. **Explainability**  
   - SHAP values for feature importance and local interpretability.

## Reproducing the Results
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run preprocessing script:
```bash
python src/preprocessing.py
```
3. Run training and evaluation:
```bash
jupyter notebook notebooks/02_preprocessing_modeling_clean.ipynb
```

## Key Findings
- Random Forest achieved the highest F1-score on evaluation data.

## Author
Daniel Sánchez Pagán
