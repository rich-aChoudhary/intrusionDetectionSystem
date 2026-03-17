import joblib
import pandas as pd
from xgboost import XGBClassifier


# ----------------------------
# Load trained models
# ----------------------------

gb_model = joblib.load("models_balanced/gradient_boosting.pkl")
ada_model = joblib.load("models_balanced/adaboost.pkl")

xgb_model = XGBClassifier()
xgb_model.load_model("models_balanced/xgboost.json")

features = joblib.load("models_balanced/feature_order.pkl")


# ----------------------------
# BENIGN samples
# ----------------------------

benign_samples = [

{
' Source Port':443,
' Destination Port':52344,
' Protocol':6,
' Flow Duration':150000,
' Total Fwd Packets':12,
' Total Backward Packets':10,
' Flow Bytes/s':1200,
' Flow Packets/s':20
},

{
' Source Port':80,
' Destination Port':49152,
' Protocol':6,
' Flow Duration':200000,
' Total Fwd Packets':20,
' Total Backward Packets':15,
' Flow Bytes/s':3000,
' Flow Packets/s':25
},

{
' Source Port':53,
' Destination Port':60000,
' Protocol':17,
' Flow Duration':80000,
' Total Fwd Packets':5,
' Total Backward Packets':4,
' Flow Bytes/s':800,
' Flow Packets/s':10
}

]


# ----------------------------
# ATTACK samples
# ----------------------------

attack_samples = [

# DDoS
{
' Source Port':80,
' Destination Port':443,
' Protocol':6,
' Flow Duration':20000,
' Total Fwd Packets':2000,
' Total Backward Packets':2,
' Flow Bytes/s':500000,
' Flow Packets/s':4000
},

# Port Scan
{
' Source Port':22,
' Destination Port':8080,
' Protocol':6,
' Flow Duration':1000,
' Total Fwd Packets':50,
' Total Backward Packets':1,
' Flow Bytes/s':20000,
' Flow Packets/s':150
},

# Bot style
{
' Source Port':4444,
' Destination Port':80,
' Protocol':6,
' Flow Duration':50000,
' Total Fwd Packets':400,
' Total Backward Packets':5,
' Flow Bytes/s':300000,
' Flow Packets/s':600
}

]


# ----------------------------
# Convert to dataframe
# ----------------------------

benign_df = pd.DataFrame(benign_samples)
attack_df = pd.DataFrame(attack_samples)

benign_df = benign_df.reindex(columns=features, fill_value=0)
attack_df = attack_df.reindex(columns=features, fill_value=0)


# ----------------------------
# Prediction function
# ----------------------------

def test_samples(df, label):

    print(f"\nTesting {label} samples\n")

    for i in range(len(df)):

        sample = df.iloc[[i]]

        gb_pred = gb_model.predict(sample)[0]
        xgb_pred = xgb_model.predict(sample)[0]
        ada_pred = ada_model.predict(sample)[0]

        print(f"Sample {i+1}")

        print("Gradient Boosting:", "ATTACK" if gb_pred==1 else "BENIGN")
        print("XGBoost:", "ATTACK" if xgb_pred==1 else "BENIGN")
        print("AdaBoost:", "ATTACK" if ada_pred==1 else "BENIGN")

        print("--------------------------------")


# ----------------------------
# Run tests
# ----------------------------

test_samples(benign_df,"BENIGN")
test_samples(attack_df,"ATTACK")