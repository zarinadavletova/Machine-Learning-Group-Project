import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 1. Load data
df = pd.read_csv("pa_features_top17.csv")

# 2. Feature and label split
X = df.drop(columns=["label"])
y = df["label"]

# 3. Encode categorical variables
X_encoded = X.copy()
for col in X_encoded.select_dtypes(include=["object"]).columns:
    X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 5. Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_probs = dt_model.predict_proba(X_test)[:, 1]
dt_report = classification_report(y_test, dt_preds, output_dict=True)
dt_auc = roc_auc_score(y_test, dt_probs)

# 6. Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_report = classification_report(y_test, rf_preds, output_dict=True)
rf_auc = roc_auc_score(y_test, rf_probs)

# 7. Compare model performance
comparison_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1-score", "ROC AUC"],
    "Decision Tree": [
        dt_report["1"]["precision"],
        dt_report["1"]["recall"],
        dt_report["1"]["f1-score"],
        dt_auc
    ],
    "Random Forest": [
        rf_report["1"]["precision"],
        rf_report["1"]["recall"],
        rf_report["1"]["f1-score"],
        rf_auc
    ]
})

# 8. Print result
print(comparison_df)
