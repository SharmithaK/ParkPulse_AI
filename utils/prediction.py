import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


#TRAIN MODEL


def train_prediction_model(df):

    features = [
        "hour",
        "vehicle_risk",
        "location_density",
        "violation_count",
        "violation_severity",
        "vehicle_type",
        "police_station",
        "violation_main"
    ]

    X = df[features]

    y = df["risk_label"]

    categorical_features = [
        "vehicle_type",
        "police_station",
        "violation_main"
    ]

    numerical_features = [
        "hour",
        "vehicle_risk",
        "location_density",
        "violation_count",
        "violation_severity"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("\n===== MISSING VALUES CHECK =====")
    print(X.isnull().sum())

    print("\n===== TARGET CHECK =====")
    print(y.isnull().sum())

    pipeline.fit(X, y)

    return pipeline

#PREDICT HOTSPOT RISK


def predict_hotspot_risk(
model,
hour,
vehicle_type,
vehicle_risk,
violation_main,
violation_count,
violation_severity,
police_station,
location_density
):

    sample = pd.DataFrame({

        "hour": [hour],

        "vehicle_risk": [vehicle_risk],

        "location_density": [location_density],

        "violation_count": [violation_count],

        "violation_severity": [violation_severity],

        "vehicle_type": [vehicle_type],

        "police_station": [police_station],

        "violation_main": [violation_main]

    })

    prediction = model.predict(sample)[0]

    probability = (
        model.predict_proba(sample)[0]
        .max()
        * 100
    )

    risk = (
        "HIGH RISK"
        if prediction == 1
        else "LOW RISK"
    )

    return {
        "risk": risk,
        "confidence": round(probability, 2)
    }