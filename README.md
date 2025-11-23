README.md
# Customer Churn Prediction

*Summary:* Predict customers likely to churn and provide retention recommendations. Includes model explainability using SHAP.

*Dataset:* Telecom churn dataset (example)  
https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset

*Files*
- notebook.ipynb
- code.py
- dataset_link.txt
- insights.txt
- interview_questions.md

  code.py

- # 📌 Churn Prediction Project
# Dataset: Telco Customer Churn (Kaggle / IBM Sample)
# ================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 2. Load Dataset
df = pd.read_csv("Telco-Customer-Churn.csv")   # Upload dataset in Colab/Local

print("Shape:", df.shape)
print(df.head())

# 3. Data Cleaning
# Remove customerID (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("Cleaned Data:\n", df.head())

# 4. Feature Scaling
X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = model.score(X_test, y_test)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {"Accuracy": acc, "ROC-AUC": auc}

    print(f"\n🔹 {name}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Compare Model Performance
results_df = pd.DataFrame(results).T
print("\nModel Performance:\n", results_df)

# Plot ROC Curve
plt.figure(figsize=(8,6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Churn Prediction")
plt.legend()
plt.show()

# 8. AI-Powered Insights
rf = models["Random Forest"]
importances = rf.feature_importances_
feature_names = X.columns
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp = feat_imp.sort_values("Importance", ascending=False)

print("\n📌 Key Drivers of Churn:\n", feat_imp.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_imp.head(10), palette="viridis")
plt.title("Top Factors Influencing Customer Churn")
plt.show()


dataset_link.txt
https://www.kaggle.com/datasets/blastchar/telco-customer-churn?utm_source=chatgpt.com

insights.txt
- Top churn drivers: tenure, monthly charges, contract type.
- Business action: offer tailored retention offers to high-probability churners.
- Use probability scores to prioritize retention budget.

interview_questions.md
Q1: Why prioritize recall in churn detection?  
A: Because catching likely churners (true positives) allows targeted retention, which reduces revenue loss.

Q2: How to measure ROI of retention campaign?  
A: Use predicted uplift vs cost of retention (LTV-based).
