# House-Price-Prediction-Advanced-Regression-
Create a robust machine learning algorithm to accurately predict the price of the house given the various factors across the market.
Overview:
This project is a house price prediction pipeline using tabular housing data (SalePrice target). The notebook performs data loading and cleaning, exploratory data analysis (distribution plots, correlations, and feature visualizations), feature engineering (age, remodeling age, total baths, porch area, log-transform of SalePrice), handling of missing and zero values, label encoding for categorical variables, scaling, and model training. Multiple regression models are trained and compared including Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost. Hyperparameter search (GridSearchCV and RandomizedSearchCV) and cross-validation are used to tune and validate models. Model evaluation reports RMSE and R^2 scores; feature importances are inspected using tree-based models.

Files:
- house prediction.ipynb  — main notebook containing EDA, preprocessing, modeling, and evaluation
- dataset (place the CSV used by the notebook) — ensure the dataset CSV is in the same directory as the notebook
- README.txt — this file

Quick start:
1. Clone or download the repository and place the dataset file alongside the notebook.
2. (Optional) Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
3. Install dependencies (example):
   pip install -r requirements.txt
4. Run the notebook:
   jupyter notebook "house prediction.ipynb"

Key steps:
- Inspect data for missing and zero-values; replace zeros in area-related columns with NaN before imputing medians.
- Encode categorical columns (LabelEncoder used in the notebook).
- Create new features: Age, RemodAge, TotalBath, TotalPorchSF, LogSalePrice.
- Train/test split (typical 80/20) and scale numeric features with StandardScaler where applicable.
- Train and compare models: Linear models (Linear, Ridge, Lasso), tree-ensemble models (RandomForest, GradientBoosting), and boosted models (XGBoost, LightGBM, CatBoost).
- Use GridSearchCV/RandomizedSearchCV for hyperparameter tuning and evaluate using RMSE and R².
- Inspect top feature importances from RandomForest.

Tips & next steps:
- If SalePrice is skewed, consider predicting log(SalePrice) and invert transform with expm1 when reporting predictions.
- Use one-hot encoding for categorical variables with many levels, or target encoding where appropriate.
- Add cross-validation and stratified folds on clusters (e.g., neighborhoods) if needed.
- Export best model with joblib or joblib.dump(best_model, 'house_model.pkl') for later use in deployment.

License:
MIT — free to use and modify for educational projects.
