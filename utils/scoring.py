import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def compute_priority_scores(df):

    priority = (
        df[(df["junction_name"] != "No Junction")&(df["junction_name"].notna())
        ].groupby("junction_name")
        .agg(
            total_violations=("id", "count"),
            avg_vehicle_risk=("vehicle_risk", "mean"),
            avg_severity=("violation_severity", "mean"),
        ).reset_index())

    priority["log_violations"] = np.log1p(priority["total_violations"])

    scaler = MinMaxScaler()

    priority[["violations_norm","vehicle_risk_norm","severity_norm",]] = scaler.fit_transform(
        priority[
            ["log_violations","avg_vehicle_risk","avg_severity",]])
    

    priority["priority_score"] = (priority["violations_norm"] * 80 + priority["severity_norm"] * 15 +priority["vehicle_risk_norm"] * 5)
    priority["priority_score"] = (priority["priority_score"]/ priority["priority_score"].max()) * 100


    priority["priority_score"] = (priority["priority_score"].round(2))

    return priority.sort_values("priority_score",ascending=False)


def attach_priority_scores(df, priority_df):

    score_map = dict(zip( priority_df["junction_name"],priority_df["priority_score"]))
    violation_map = dict(zip(priority_df["junction_name"], priority_df["total_violations"]))
    severity_map = dict(zip(priority_df["junction_name"],priority_df["avg_severity"]))
    vehicle_map = dict( zip(priority_df["junction_name"],priority_df["avg_vehicle_risk"]))

    df["priority_score"] = (df["junction_name"].map(score_map).fillna(0))
    df["junction_total_violations"] = (df["junction_name"].map(violation_map).fillna(0))
    df["junction_avg_severity"] = (df["junction_name"].map(severity_map).fillna(0))
    df["junction_avg_vehicle_risk"] = (df["junction_name"].map(vehicle_map).fillna(0))
    return df


def create_risk_label(df,priority_df):

    threshold = priority_df["priority_score"].quantile(0.75)

    df["risk_label"] = ( df["priority_score"] >= threshold).astype(int)
    return df


def scoring_pipeline(df):

    priority_df = compute_priority_scores(df)

    df = attach_priority_scores(df, priority_df)
    df = create_risk_label(df,priority_df)

    return df, priority_df
