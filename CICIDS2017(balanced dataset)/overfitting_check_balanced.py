import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
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


print("Loading trained models...")

gb_model = joblib.load("models_balanced/gradient_boosting.pkl")
ada_model = joblib.load("models_balanced/adaboost.pkl")

xgb_model = XGBClassifier()
xgb_model.load_model("models_balanced/xgboost.json")


def check_overfitting(model, name):

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("\n", name)
    print("Training Accuracy :", train_acc)
    print("Testing Accuracy  :", test_acc)
    print("Difference        :", train_acc - test_acc)


print("\n------ Overfitting Check (Balanced Dataset) ------")

check_overfitting(gb_model,"Gradient Boosting")
check_overfitting(xgb_model,"XGBoost")
check_overfitting(ada_model,"AdaBoost")