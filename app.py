import streamlit as st
from datetime import datetime
from utils.preprocessing import (load_raw_data,preprocess_pipeline)
from utils.scoring import (scoring_pipeline)
from utils.recommendation import (get_recommendation,generate_recommendations)
from utils.prediction import (train_prediction_model,predict_hotspot_risk)

st.set_page_config(page_title="ParkPulseAI",layout="wide")

st.title("ParkPulseAI")
st.caption("AI-powered Parking Hotspot Detection & Enforcement Intelligence System")

@st.cache_data
def load_data():
    df = load_raw_data("data/jan to may police violation_anonymized791b166.csv")
    df = preprocess_pipeline(df)
    df, priority_df = scoring_pipeline(df)
    return df, priority_df
df, priority_df = load_data()

@st.cache_resource
def get_model(df):
    return train_prediction_model(df)

model = get_model(df)

page = st.sidebar.radio(
"Navigation",[
"Executive Dashboard",
"Heatmap",
"Violation Analytics",
"Time Pattern Analysis",
"AI Priority Score",
"Recommendation Engine",
"Hotspot Prediction"
])


if page == "Executive Dashboard":

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Violations",len(df))
    c2.metric("Police Stations",df["police_station"].nunique())
    c3.metric("Junctions",df["junction_name"].nunique())

    st.subheader("Most Common Violation")
    st.write(df["violation_main"].value_counts().head(5))

    st.subheader("Most Violating Vehicle")
    st.write(df["vehicle_type"].value_counts().head(5))


elif page == "Heatmap":
    st.subheader("Parking Hotspots")
    st.map(df[["latitude", "longitude"]].dropna())

elif page == "Violation Analytics":
    st.subheader("Violation Distribution")
    st.bar_chart(df["violation_main"].value_counts())

    st.subheader("Vehicle Distribution")
    st.bar_chart( df["vehicle_type"].value_counts())

    st.subheader("Top Police Stations")
    st.bar_chart( df["police_station"].value_counts() .head(10))


elif page == "Time Pattern Analysis":
    st.subheader("Hourly Violations")
    st.line_chart( df.groupby("hour").size())

    st.subheader("Daily Violations")
    st.bar_chart( df["day"].value_counts())

    st.subheader("Monthly Violations")
    st.bar_chart( df["month"].value_counts())


elif page == "AI Priority Score":
    st.subheader("Top Risk Junctions")
    st.dataframe( priority_df.head(20))


elif page == "Recommendation Engine":

   st.subheader(" AI Recommendation Engine")

   top = priority_df.head(10)
   for _, row in top.iterrows():

        rec = get_recommendation(row["priority_score"])
        st.markdown("---")
        st.success(f"{row['junction_name']}")

        st.metric("Priority Score",round(row["priority_score"], 2))

        st.error(f"Risk Level: {rec['level']}")

        st.write(f" Recommended Action:  {rec['action']}")
        st.write(f" Monitoring Strategy:  {rec['monitoring']}")
        st.write(f" Enforcement Priority:  {rec['priority']}")

elif page == "Hotspot Prediction":

    st.subheader( "Predict Future Violation Risk")

    junction = st.selectbox("Junction",sorted( priority_df["junction_name"]))

    day = st.selectbox(
    "Day",["Monday","Tuesday","Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])


    hour = st.slider("Hour", 0, 23,9)

    if st.button("Predict Risk"):

        station = (df[df["junction_name"] == junction]["police_station"].mode()[0])

        junction_df = df[df["junction_name"] == junction]
        vehicle_risk = junction_df["vehicle_risk"].mode()[0]
        violation_count = junction_df["violation_count"].mode()[0]
        violation_severity = round( junction_df["violation_severity"].mean(), 2)

        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
        is_peak_hour = 1 if ( 8 <= hour <= 11 or 17 <= hour <= 20) else 0
        month=month = datetime.now().strftime("%B")

        result = predict_hotspot_risk(
                model=model,
                hour=hour,
                day=day,
                month=month,
                vehicle_risk=vehicle_risk,
                violation_count=violation_count,
                violation_severity=violation_severity,
                junction_name=junction,
                police_station=station,
                is_weekend=is_weekend,
                is_peak_hour=is_peak_hour)
        
        chance = result["high_risk_probability"]
        if chance >= 80:
            level = "Critical"
            action = "Immediate patrol deployment recommended"

        elif chance >= 60:
                level = " High"
                action = "Increase patrol frequency"

        elif chance >= 40:
                level = "Moderate"
                action = "Monitor during peak hours"

        else:
                level = "Low"
                action = "Routine monitoring sufficient"

        st.metric("Future High-Risk Probability",f"{chance}%")
        st.error(f"Risk Level: {level}")
        st.success(f"Nearest Police Station: {station}")
        st.info(f" Recommended Action: {action}")

        st.markdown("### Top violation")

        top_violations = (
            df[df["junction_name"] == junction]["violation_main"].value_counts().head(3))

        for v, count in top_violations.items():
            st.write(f"• {v} ({count})")