# ParkPulseAI

### AI-Powered Parking Hotspot Detection & Enforcement Intelligence System
TriVars
## Overview

ParkPulseAI is an AI-driven parking intelligence platform designed to identify illegal parking hotspots, analyze parking violation patterns, predict future risk, and recommend targeted enforcement strategies.
The system transforms historical traffic violation data into actionable insights for traffic police and smart city authorities, enabling proactive parking management instead of reactive patrol-based enforcement.

---

## Problem Statement

Illegal parking near commercial areas, metro stations, schools, hospitals, and busy junctions causes congestion and disrupts traffic flow.

Current enforcement methods are:
* Reactive and patrol-based
* Difficult to prioritize high-risk zones
* Lacking hotspot visibility
* Unable to predict future parking risks

ParkPulseAI addresses these challenges using data analytics and machine learning.

---

## Features

### Executive Dashboard

* Total violations
* Number of police stations
* Number of junctions
* Most common violations
* Vehicle-wise violation analysis

### Heatmap Visualization

* Displays geographical distribution of parking violations
* Identifies congestion hotspots

### Violation Analytics

* Violation distribution
* Vehicle distribution
* Top police stations

### Time Pattern Analysis

* Hourly trends
* Daily patterns
* Monthly patterns
* Peak hour analysis

### AI Priority Score

Ranks junctions based on:

* Violation volume
* Violation severity
* Vehicle risk

### Recommendation Engine

Provides:
* Risk level
* Enforcement priority
* Monitoring strategy
* Recommended actions

### Future Risk Prediction

Predicts probability of future high-risk violations using:
* Hour
* Day
* Month
* Vehicle risk
* Violation count
* Violation severity
* Peak hour indicator
* Weekend indicator
* Junction
* Police station

Outputs:

* Future high-risk probability
* Risk level
* Nearest police station
* Recommended action
* Top 3 violations at the selected junction

---

## Feature Engineering

### Time Features

* Hour
* Day
* Month
* Weekday
* Weekend flag
* Peak hour flag

### Violation Features

* Main violation extraction
* Violation count
* Severity scoring

### Vehicle Features

Risk scores assigned based on vehicle category:

* Two-wheelers
* Cars
* Autos
* Buses
* Heavy vehicles

---

## Machine Learning

### Algorithm

Random Forest Classifier

### Target

High-risk hotspot classification

### Outputs

* High-risk probability (%)
* Risk level
* Enforcement recommendations

---

## Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* Random Forest Classifier
* Pipeline
* ColumnTransformer
* OneHotEncoder

---

## Folder Structure

```
ParkPulseAI
│
├── app.py
│
├── data
│
├── utils
│   ├── preprocessing.py
│   ├── scoring.py
│   ├── prediction.py
│   └── recommendation.py
│── requirement.txt
└── README.md

```

---

## Future Scope

* Real-time CCTV integration
* Congestion impact estimation
* Dynamic patrol optimization
* Explainable AI
* Live traffic integration
* Smart city deployment

---

## Impact

ParkPulseAI enables proactive parking enforcement by identifying hotspots, predicting future risks, and helping authorities allocate resources efficiently.

### Making cities smarter, safer and congestion-free.
