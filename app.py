import streamlit as st

from utils.preprocessing import (
load_raw_data,
preprocess_pipeline
)

from utils.scoring import (
scoring_pipeline
)

from utils.recommendation import (
get_recommendation,
generate_recommendations
)

from utils.prediction import (
train_prediction_model,
predict_hotspot_risk
)

# =====================================================
# #PAGE CONFIG
# =====================================================

st.set_page_config(
page_title="ParkPulseAI",
layout="wide"
)

st.title("🚔 ParkPulseAI")
st.caption(
"AI-powered parking intelligence system"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = load_raw_data(
        "data/jan to may police violation_anonymized791b166.csv"
    )

    df = preprocess_pipeline(df)

    df, priority_df = scoring_pipeline(df)

    return df, priority_df

df, priority_df = load_data()

# =====================================================
# TRAIN MODEL
# =====================================================

@st.cache_resource
def get_model(df):
    return train_prediction_model(df)

model = get_model(df)

# =====================================================
# SIDEBAR
# =====================================================

page = st.sidebar.radio(
"Navigation",
[
"Executive Dashboard",
"Heatmap",
"Violation Analytics",
"Time Pattern Analysis",
"AI Priority Score",
"Recommendation Engine",
"Hotspot Prediction"
]
)

# =====================================================
# EXECUTIVE DASHBOARD
# =====================================================

if page == "Executive Dashboard":

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Violations",
        len(df)
    )

    c2.metric(
        "Police Stations",
        df["police_station"].nunique()
    )

    c3.metric(
        "Junctions",
        df["junction_name"].nunique()
    )

    st.subheader("Most Common Violation")

    st.write(
        df["violation_main"]
        .value_counts()
        .head(5)
    )

    st.subheader("Most Violating Vehicle")

    st.write(
        df["vehicle_type"]
        .value_counts()
        .head(5)
    )
# =====================================================
# HEATMAP
# =====================================================

elif page == "Heatmap":

    st.subheader("Parking Hotspots")

    st.map(
        df[
            ["latitude", "longitude"]
        ].dropna()
    )
# =====================================================
# VIOLATION ANALYTICS
# =====================================================

elif page == "Violation Analytics":

    st.subheader("Violation Distribution")

    st.bar_chart(
        df["violation_main"]
        .value_counts()
    )

    st.subheader("Vehicle Distribution")

    st.bar_chart(
        df["vehicle_type"]
        .value_counts()
    )

    st.subheader("Top Police Stations")

    st.bar_chart(
        df["police_station"]
        .value_counts()
        .head(10)
    )
# =====================================================
# TIME ANALYSIS
# =====================================================

elif page == "Time Pattern Analysis":

    st.subheader("Hourly Violations")

    st.line_chart(
        df.groupby("hour")
        .size()
    )

    st.subheader("Daily Violations")

    st.bar_chart(
        df["day"]
        .value_counts()
    )

    st.subheader("Monthly Violations")

    st.bar_chart(
        df["month"]
        .value_counts()
    )
# =====================================================
# PRIORITY SCORE
# =====================================================

elif page == "AI Priority Score":

    st.subheader("Top Risk Junctions")

    st.dataframe(
        priority_df.head(20)
    )
# =====================================================
# RECOMMENDATION ENGINE
# =====================================================

elif page == "Recommendation Engine":

   st.subheader("🚨 AI Recommendation Engine")

   top = priority_df.head(10)

   for _, row in top.iterrows():

        rec = get_recommendation(
            row["priority_score"]
        )

        st.markdown("---")

        st.success(
            f"📍 {row['junction_name']}"
        )

        st.metric(
            "Priority Score",
            round(row["priority_score"], 2)
        )

        st.error(
            f"Risk Level: {rec['level']}"
        )

        st.write(
            f"🚔 Recommended Action: {rec['action']}"
        )

        st.write(
            f"📡 Monitoring Strategy: {rec['monitoring']}"
        )

        st.write(
            f"⚡ Enforcement Priority: {rec['priority']}"
        )
# =====================================================
# HOTSPOT PREDICTION
# =====================================================

elif page == "Hotspot Prediction":

    st.subheader(
        "Predict Future Violation Risk"
    )

    station = st.selectbox(
        "Police Station",
        sorted(
            df["police_station"]
            .dropna()
            .unique()
        )
    )

    vehicle = st.selectbox(
        "Vehicle Type",
        sorted(
            df["vehicle_type"]
            .dropna()
            .unique()
        )
    )

    violation = st.selectbox(
        "Violation",
        sorted(
            df["violation_main"]
            .dropna()
            .unique()
        )
    )

    hour = st.slider(
        "Hour",
        0,
        23,
        9
    )

    if st.button("Predict Risk"):

        vehicle_risk = (
            df[df["vehicle_type"] == vehicle]
            ["vehicle_risk"]
            .mode()[0]
        )

        violation_severity = (
            df[df["violation_main"] == violation]
            ["violation_severity"]
            .mode()[0]
        )

        density = (
            df[df["police_station"] == station]
            ["location_density"]
            .mean()
        )

        result = predict_hotspot_risk(
            model=model,
            hour=hour,
            vehicle_type=vehicle,
            vehicle_risk=vehicle_risk,
            violation_main=violation,
            violation_count=1,
            violation_severity=violation_severity,
            police_station=station,
            location_density=density
        )

        st.metric(
            "Risk",
            result["risk"]
        )

        st.metric(
            "Confidence %",
            result["confidence"]
        )