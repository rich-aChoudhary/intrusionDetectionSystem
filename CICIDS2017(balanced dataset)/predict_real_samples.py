import joblib
import pandas as pd
from xgboost import XGBClassifier

print("Loading dataset...")

dataset = pd.read_csv("balanced_dataset.csv")

X = dataset.drop(columns=[' Label'])
y = dataset[' Label']

# load models
gb_model = joblib.load("models_balanced/gradient_boosting.pkl")
ada_model = joblib.load("models_balanced/adaboost.pkl")

xgb_model = XGBClassifier()
xgb_model.load_model("models_balanced/xgboost.json")


# -------------------------
# Select real samples
# -------------------------

benign_samples = X[y == 0].sample(3, random_state=1)
attack_samples = X[y == 1].sample(3, random_state=1)


def test_samples(samples, label):

    print(f"\nTesting REAL {label} samples\n")

    for i in range(len(samples)):

        sample = samples.iloc[[i]]

        gb_pred = gb_model.predict(sample)[0]
        xgb_pred = xgb_model.predict(sample)[0]
        ada_pred = ada_model.predict(sample)[0]

        print(f"Sample {i+1}")

        print("Gradient Boosting:", "ATTACK" if gb_pred==1 else "BENIGN")
        print("XGBoost:", "ATTACK" if xgb_pred==1 else "BENIGN")
        print("AdaBoost:", "ATTACK" if ada_pred==1 else "BENIGN")

        print("------------------------")


test_samples(benign_samples, "BENIGN")
test_samples(attack_samples, "ATTACK")