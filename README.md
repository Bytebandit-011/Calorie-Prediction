# Calorie Prediction Project

## Overview
This project implements a machine learning model to predict calorie burn based on various features. The model uses linear regression with feature scaling to make predictions on exercise-related data.

## Dataset
- **Source**: Kaggle Playground Series S5E5
- **Files**: 
  - `train.csv` - Training dataset with features and target variable (Calories)
  - `test.csv` - Test dataset for making predictions
- **Target Variable**: Calories burned
- **Features**: Include demographic and exercise-related variables (with 'Sex' being one categorical feature)

## Project Structure
```
├── train.csv                 # Training data
├── test.csv                  # Test data for predictions
├── submission2.csv           # Final predictions output
└── main.py                   # Main analysis and modeling script
```

## Methodology

### 1. Data Preprocessing
- **Categorical Encoding**: One-hot encoding applied to 'Sex' column
- **Feature Selection**: Removed 'id' column and separated target variable 'Calories'
- **Data Splitting**: 70% training, 30% validation split with random_state=42

### 2. Feature Scaling
- **Standard Scaler**: Applied to normalize features (zero mean, unit variance)
- Prevents features with larger scales from dominating the model
- Applied consistently to training, validation, and test sets

### 3. Model Training
- **Algorithm**: Linear Regression
- **Training**: Fitted on scaled training data
- **Prediction Clipping**: Applied non-negativity constraint (calories ≥ 0)

### 4. Model Evaluation
- **Metric**: RMSLE (Root Mean Squared Logarithmic Error)
- Suitable for regression problems where relative errors matter more than absolute errors
- Handles the non-negative nature of calorie predictions well

## Key Features
- **Robust Preprocessing**: Handles categorical variables and scaling
- **Validation**: Proper train-test split for model evaluation
- **Constraint Handling**: Ensures non-negative calorie predictions
- **Scalable Pipeline**: Easy to extend with additional features or models

## Installation & Requirements
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
```

## Usage
1. **Data Loading**: Place `train.csv` and `test.csv` in the appropriate directory
2. **Run the Script**: Execute the main analysis script
3. **Output**: Generates `submission2.csv` with predictions for the test set

## Model Performance
The model is evaluated using RMSLE (Root Mean Squared Logarithmic Error), which is printed during execution. This metric is particularly appropriate for:
- Non-negative target variables
- Cases where relative errors are more important than absolute errors
- Competition submissions where RMSLE is the standard metric

## Output Format
The final submission file (`submission2.csv`) contains:
- `id`: Unique identifier for each test sample
- `Calories`: Predicted calorie burn values (non-negative)

## Future Improvements
- **Feature Engineering**: Create interaction terms or polynomial features
- **Model Selection**: Try ensemble methods (Random Forest, XGBoost)
- **Hyperparameter Tuning**: Optimize model parameters using cross-validation
- **Feature Selection**: Use techniques like recursive feature elimination
- **Advanced Validation**: Implement k-fold cross-validation for better performance estimates

## Notes
- All predictions are clipped to ensure non-negative values
- The model uses a simple linear regression approach, which provides a good baseline
- Feature scaling is crucial for linear regression performance
- The random state ensures reproducible results

## License
This project is part of a Kaggle competition and follows standard data science practices for educational and competitive purposes.
