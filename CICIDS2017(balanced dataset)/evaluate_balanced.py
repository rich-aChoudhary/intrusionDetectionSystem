import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
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


# -----------------------------
# Load models
# -----------------------------

gb_model = joblib.load("models_balanced/gradient_boosting.pkl")
ada_model = joblib.load("models_balanced/adaboost.pkl")

xgb_model = XGBClassifier()
xgb_model.load_model("models_balanced/xgboost.json")


# -----------------------------
# Evaluation function
# -----------------------------

def evaluate(model, name):

    pred = model.predict(X_test)

    print("\n", name)

    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1-score :", f1_score(y_test, pred))

    print("\nConfusion Matrix")

    print(confusion_matrix(y_test, pred))


evaluate(gb_model, "Gradient Boosting")
evaluate(xgb_model, "XGBoost")
evaluate(ada_model, "AdaBoost")