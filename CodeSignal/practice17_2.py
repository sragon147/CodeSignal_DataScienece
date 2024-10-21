import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.svm import SVR

# Load the Boston housing dataset
boston = fetch_openml(name='boston', version=1, as_frame=True)
data = boston.frame
X = data.drop(columns='MEDV')  # Assuming 'MEDV' is the correct target column name
y = data['MEDV']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical and categorical features
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
category_features = X.select_dtypes(include=['object']).columns

# Define preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

category_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', category_transformer, category_features)
    ]
)

# Define the parameter grid to search
param_grid = {
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Create a pipeline that combines preprocessing with RandomForestRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=42))
])

# Setup cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform grid search on training data
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters found: ", grid_search.best_params_)
print("Best Negative Mean Squared Error: {:.2f}".format(grid_search.best_score_ * -1))

# Predict on the test set with best parameters
best_rf_reg = grid_search.best_estimator_
y_pred = best_rf_reg.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = best_rf_reg.score(X_test, y_test)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')
print(f'Accuracy: {accuracy:.2f}')

# Get feature importances and convert to DataFrame
importances = best_rf_reg.named_steps['rf'].feature_importances_
feature_names = list(numerical_features) + list(category_features)  # Combine numerical and categorical feature names

importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort features by importance
importances_df = importances_df.sort_values(by='Importance', ascending=False)

top_n = 5
top_features = importances_df['Feature'][:top_n]

print(f'Top {top_n} features: {top_features.values}')
print(X_train.head())


# Select top features from X_train and X_test
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

numerical_features_top = X_train_top.select_dtypes(include=['float64', 'int64']).columns
category_features_top = X_train_top.select_dtypes(include=['object']).columns

# Combine preprocessing steps into a ColumnTransformer
preprocessor_top = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_top),
        ('cat', category_transformer, category_features_top)
    ]
)

# Create a new pipeline for top features
pipeline_top = Pipeline(steps=[
    ('preprocessor', preprocessor_top),
    ('rf', RandomForestRegressor(random_state=42))
])

# Define the parameter grid to search for top features
param_grid_top = {
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Instantiate GridSearchCV for top features
grid_search_top = GridSearchCV(estimator=pipeline_top, param_grid=param_grid_top, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform grid search on training data with top features
grid_search_top.fit(X_train_top, y_train)

# Best parameters and best score with top features
print("Best Parameters found with top features: ", grid_search_top.best_params_)
print("Best Negative Mean Squared Error with top features: {:.2f}".format(grid_search_top.best_score_ * -1))

# Predict on the test set with best parameters and top features
best_rf_reg_top = grid_search_top.best_estimator_
y_pred_top = best_rf_reg_top.predict(X_test_top)

# Evaluating the model with top features
mse_top = mean_squared_error(y_test, y_pred_top)
r2_top = r2_score(y_test, y_pred_top)
accuracy_top = best_rf_reg_top.score(X_test_top, y_test)
print(f'Mean Squared Error with top {top_n} features: {mse_top:.2f}')
print(f'R^2 Score with top {top_n} features: {r2_top:.2f}')
print(f'Accuracy with top {top_n} features: {accuracy_top:.2f}')
