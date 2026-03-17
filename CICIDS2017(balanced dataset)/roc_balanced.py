import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier


print("Loading balanced dataset...")

dataset = pd.read_csv("balanced_dataset.csv")

X = dataset.drop(columns=[' Label'])
y = dataset[' Label']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)


print("Loading trained models...")

gb_model = joblib.load("models_balanced/gradient_boosting.pkl")
ada_model = joblib.load("models_balanced/adaboost.pkl")

xgb_model = XGBClassifier()
xgb_model.load_model("models_balanced/xgboost.json")


print("Generating ROC curves...")


gb_probs = gb_model.predict_proba(X_test)[:,1]
xgb_probs = xgb_model.predict_proba(X_test)[:,1]
ada_probs = ada_model.predict_proba(X_test)[:,1]


fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
fpr_ada, tpr_ada, _ = roc_curve(y_test, ada_probs)


auc_gb = auc(fpr_gb, tpr_gb)
auc_xgb = auc(fpr_xgb, tpr_xgb)
auc_ada = auc(fpr_ada, tpr_ada)


plt.figure(figsize=(8,6))

plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC = {auc_gb:.4f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.4f})")
plt.plot(fpr_ada, tpr_ada, label=f"AdaBoost (AUC = {auc_ada:.4f})")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison (Balanced Dataset)")

plt.legend()
plt.grid()

plt.savefig("roc_balanced.png", dpi=300)

plt.show()

print("\nROC curve saved as roc_balanced.png")