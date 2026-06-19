import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def train_prediction_model(df):

    features =["hour","day","vehicle_risk","violation_count","violation_severity",
    "is_weekend","is_peak_hour","junction_name","police_station","month"]

    X = df[features]
    y = df["risk_label"]

    categorical_features = ["day","month","junction_name","police_station"
]

    numerical_features = ["hour","vehicle_risk","violation_count","violation_severity","is_weekend",
    "is_peak_hour",]

    preprocessor = ColumnTransformer(transformers=[
            ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)],remainder="passthrough")

    model = RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1,   max_depth=15,
    min_samples_leaf=5, class_weight="balanced")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)])
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)

    pipeline.fit(X_train, y_train)

    print("Test Accuracy =", pipeline.score(X_test, y_test))
    return pipeline

def predict_hotspot_risk(model,hour,vehicle_risk,violation_count,violation_severity,day,month,
    junction_name,
    police_station, is_weekend,
    is_peak_hour):

    sample = pd.DataFrame({
        "hour": [hour],
        "vehicle_risk": [vehicle_risk],
        "day":[day],
        "month":[month],
        "violation_count": [violation_count],
        "violation_severity": [violation_severity],"is_weekend":[is_weekend],
        "is_peak_hour":[is_peak_hour],
        "junction_name":[junction_name],
        "police_station":[police_station]})
    
    probs = model.predict_proba(sample)
    high_risk_probability = probs[0][1]*100
    

    return {
    "high_risk_probability": round(high_risk_probability, 2)
    }
