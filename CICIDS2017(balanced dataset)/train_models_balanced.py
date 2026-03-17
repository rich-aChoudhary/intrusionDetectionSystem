import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ----------------------------
# Load balanced dataset
# ----------------------------

print("Loading balanced dataset...")

dataset = pd.read_csv("balanced_dataset.csv")

print("Dataset shape:", dataset.shape)

X = dataset.drop(columns=[' Label'])
y = dataset[' Label']


# ----------------------------
# Train test split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ----------------------------
# Create model folder
# ----------------------------

os.makedirs("models_balanced", exist_ok=True)

# save feature order
joblib.dump(X_train.columns.tolist(), "models_balanced/feature_order.pkl")


# ============================
# Gradient Boosting
# ============================

gb_path = "models_balanced/gradient_boosting.pkl"

if os.path.exists(gb_path):

    print("\nLoading existing Gradient Boosting model...")
    gb_model = joblib.load(gb_path)

else:

    print("\nTraining Gradient Boosting model...")

    gb_model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    joblib.dump(gb_model, gb_path)

    print("Gradient Boosting model saved.")


# ============================
# XGBoost
# ============================

xgb_path = "models_balanced/xgboost.json"

if os.path.exists(xgb_path):

    print("\nLoading existing XGBoost model...")
    xgb_model = XGBClassifier()
    xgb_model.load_model(xgb_path)

else:

    print("\nTraining XGBoost model...")

    xgb_model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)

    xgb_model.save_model(xgb_path)

    print("XGBoost model saved.")


# ============================
# AdaBoost
# ============================

ada_path = "models_balanced/adaboost.pkl"

if os.path.exists(ada_path):

    print("\nLoading existing AdaBoost model...")
    ada_model = joblib.load(ada_path)

else:

    print("\nTraining AdaBoost model...")

    ada_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    ada_model.fit(X_train, y_train)

    joblib.dump(ada_model, ada_path)

    print("AdaBoost model saved.")


print("\nAll balanced models ready.")