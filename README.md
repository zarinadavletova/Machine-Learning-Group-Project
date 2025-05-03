# Machine Learning Project: Predicting Project Funding Success

## Overview

This project uses real-world data to predict whether education-related projects will be fully funded. The pipeline involves data cleaning, feature engineering, splitting, model building (baseline and advanced), and evaluation. The goal is to develop interpretable and accurate machine learning models to support decision-making.

---

## Project Structure

### 📁 Notebooks

- **`description_datasets_ML_Project.ipynb`**
  - Provides an exploratory overview of all datasets used in the project.
  - Includes summary statistics and initial observations.

- **`data cleaning.ipynb`**
  - Handles missing values, incorrect formats, and feature renaming.
  - Prepares the raw data for further processing.

---
### 📁 Python Scripts

- **`feature_slection.py`**
  - Selects relevant features 
  
- **`data splitting.py`**
  - Splits the feature matrix into training, validation, and test sets.

- **`baseline model.py`**
  - Trains and evaluates a simple baseline model 
  - Serves as a benchmark for comparison with more advanced models.

- **`advanced model.py`**
  - Implements advanced machine learning algorithms  Random Forest, XGBoost.
  - Includes parameter tuning and performance evaluation.

---

## Workflow Summary

1. **Data Exploration**: Understand and visualize the datasets.
2. **Cleaning & Preprocessing**: Clean the data and create features.
3. **Feature Engineering**: Aggregate resource and donation info.
4. **Feature Selection**: Identify the most predictive variables.
5. **Data Splitting**: Separate data for training, validation, and testing.
6. **Modeling**:
   - Build a baseline model.
   - Develop and evaluate more complex models.
7. **Evaluation**: Compare models and choose the best-performing one.

---
## Requirements

 - pandas
 - numpy 
 - matplotlib.pyplot 
 - seaborn 
 - RandomForestClassifier
 - XGBClassifier
 - SimpleImputer
 - Pipeline
 - OneHotEncoder, StandardScaler
 - ColumnTransformer
 - classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
 - GridSearchCV, StratifiedKFold
 - train_test_split
 - LabelEncoder, StandardScaler
 - LogisticRegression
 - RandomForestClassifier
 - VarianceThreshold
 - os


