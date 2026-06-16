import pandas as pd
from sklearn.preprocessing import MinMaxScaler


#CALCULATE PRIORITY SCORE

def compute_priority_scores(df):

# Aggregate by junction

    priority = (
        df.groupby("junction_name")
        .agg(
            total_violations=("id", "count"),
            avg_vehicle_risk=("vehicle_risk", "mean"),
            avg_severity=("violation_severity", "mean"),
            location_density=("location_density", "mean")
        )
        .reset_index()
    )

# -----------------------------------------
# Normalize metrics
# -----------------------------------------

    scaler = MinMaxScaler()

    priority[
        [
            "violations_norm",
            "vehicle_risk_norm",
            "severity_norm",
            "density_norm"
        ]
    ] = scaler.fit_transform(
        priority[
            [
                "total_violations",
                "avg_vehicle_risk",
                "avg_severity",
                "location_density"
            ]
        ]
    )

# -----------------------------------------
# AI Priority Score
#
# 40% violation volume
# 25% vehicle risk
# 20% location density
# 15% severity
# -----------------------------------------

    priority["priority_score"] = (

        priority["violations_norm"] * 40 +

        priority["vehicle_risk_norm"] * 25 +

        priority["density_norm"] * 20 +

        priority["severity_norm"] * 15

    )

    priority["priority_score"] = (
        priority["priority_score"]
        .round(2)
    )

    return priority.sort_values(
        "priority_score",
        ascending=False
    )

#MAP SCORES BACK TO ROW LEVEL


def attach_priority_scores(df, priority_df):

    score_map = dict(
        zip(
            priority_df["junction_name"],
            priority_df["priority_score"]
        )
    )

    df["priority_score"] = (
        df["junction_name"]
        .map(score_map)
    )

    return df

#CREATE RISK LABEL

def create_risk_label(df):

    df["risk_label"] = (
        df["priority_score"] >= 70
    ).astype(int)

    return df

#COMPLETE AI SCORING PIPELINE


def scoring_pipeline(df):

    priority_df = compute_priority_scores(df)

    df = attach_priority_scores(
        df,
        priority_df
    )

    df = create_risk_label(df)

    return df, priority_df