def get_recommendation(score):

    if score >= 85:
        return {
            "level": "CRITICAL",
            "action": "Deploy additional officers immediately",
            "monitoring": "24x7 hotspot monitoring",
            "priority": "Highest"
        }

    elif score >= 70:
        return {
            "level": "HIGH",
            "action": "Increase patrol frequency",
            "monitoring": "Peak-hour monitoring required",
            "priority": "High"
        }

    elif score >= 50:
        return {
            "level": "MEDIUM",
            "action": "Schedule regular enforcement checks",
            "monitoring": "Monitor during busy hours",
            "priority": "Moderate"
        }

    else:
        return {
            "level": "LOW",
            "action": "Routine monitoring sufficient",
            "monitoring": "No additional deployment required",
            "priority": "Low"
        }

#BULK RECOMMENDATIONS


def generate_recommendations(priority_df):

    recommendations = []

    for _, row in priority_df.iterrows():

        rec = get_recommendation(
            row["priority_score"]
        )

        recommendations.append({
            "junction_name": row["junction_name"],
            "priority_score": row["priority_score"],
            "risk_level": rec["level"],
            "action": rec["action"],
            "monitoring": rec["monitoring"],
            "priority": rec["priority"]
        })

    return recommendations