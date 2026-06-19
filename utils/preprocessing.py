import pandas as pd
import numpy as np
import ast


def load_raw_data(path):
    return pd.read_csv(path, low_memory=False)


def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates(subset=["id"])

    cols_to_drop = ["description","closed_datetime","action_taken_timestamp"]

    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_cols, inplace=True)


    df["location"] = df["location"].fillna("Unknown")
    df["police_station"] = (df["police_station"].fillna("Unknown"))

    df["junction_name"] = (df["junction_name"].fillna("No Junction"))

    df["validation_status"] = (df["validation_status"].fillna("pending"))

    df["center_code"] = (df["center_code"].fillna(df["center_code"].median()))


    date_cols = [
        "created_datetime",
        "modified_datetime",
        "data_sent_to_scita_timestamp",
        "validation_timestamp"
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col],errors="coerce")
    return df


def create_time_features(df):

    df["hour"] = (df["created_datetime"].dt.hour.fillna(0).astype(int))
    df["day"] = (df["created_datetime"].dt.day_name()).fillna("Unknown")
    df["month"] = (df["created_datetime"].dt.month_name())
    df["weekday"] = (df["created_datetime"].dt.weekday)
    df["is_weekend"] = (df["day"].isin(["Saturday","Sunday"])).astype(int)
    df["is_peak_hour"] = ((df["hour"].between(8,11)) |(df["hour"].between(17,20))).astype(int)
    return df

def create_violation_features(df):

    def get_main_violation(x):
        try:
            vals = ast.literal_eval(x)
            return vals[0]
        except:
            return "UNKNOWN"

    def get_violation_count(x):
        try:
            vals = ast.literal_eval(x)
            return len(vals)
        except:
            return 1

    severity_map = {
        "PARKING NEAR ROAD CROSSING": 5,
        "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 5,

        "PARKING IN A MAIN ROAD": 4,
        "DOUBLE PARKING": 4,
        "PARKING ON FOOTPATH": 4,
        "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 4,

        "NO PARKING": 3,

        "WRONG PARKING": 2,

        "DEFECTIVE NUMBER PLATE": 2,
        "USING BLACK FILM/OTHER MATERIALS": 2,

        "DEMANDING EXCESS FARE": 1,
        "REFUSE TO GO FOR HIRE": 1
    }

    df["violation_main"] = (df["violation_type"].apply(get_main_violation))
    df["violation_count"] = ( df["violation_type"].apply(get_violation_count))

    df["violation_severity"] = (df["violation_main"].map(severity_map).fillna(2))

    return df

def create_vehicle_features(df):

    vehicle_risk_map = {
        "SCOOTER": 1,"MOPED": 1,"MOTOR CYCLE": 1,
        "PASSENGER AUTO": 2,"CAR": 2,"JEEP": 2,"VAN": 2,"MAXI-CAB": 2,
        "GOODS AUTO": 3,"TEMPO": 3,"LGV": 3,
        "BUS (BMTC/KSRTC)": 4,"PRIVATE BUS": 4,"TOURIST BUS": 4,"SCHOOL VEHICLE": 4,"FACTORY BUS": 4,
        "HGV": 5,"LORRY/GOODS VEHICLE": 5,"TANKER": 5,"TRACTOR": 5,"MINI LORRY": 5,
        "OTHERS": 2
    }

    df["vehicle_risk"] = (df["vehicle_type"].map(vehicle_risk_map).fillna(2))

    return df


def clean_geodata(df):

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(12, 14)) &(df["longitude"].between(76, 78))]
    return df

def preprocess_pipeline(df):
    df = clean_data(df)
    df = clean_geodata(df)
    df = create_time_features(df)
    df = create_violation_features(df)
    df = create_vehicle_features(df)
    return df

