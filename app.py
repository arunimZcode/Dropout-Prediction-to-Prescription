
import pickle, pandas as pd, os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open("dropout_model.pkl", "rb") as f:
    model_bundle = pickle.load(f)
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

model     = model_bundle["model"]
threshold = model_bundle["threshold"]
scaler    = preprocessor["scaler"]
encoders  = preprocessor["encoders"]
feat_cols = preprocessor["feature_cols"]

SUGGESTIONS = {
    "Support_Deficit": {
        "student": ("Support", "You currently have limited support at school and home. Reach out to your school counsellor — they can connect you with resources and people who want to help you succeed."),
        "faculty": ("Support network", "This student lacks both school and family support. Assign a trusted staff mentor and check in weekly.")
    },
    "Study_Time": {
        "student": ("Study habits", "Try dedicating at least 1 hour each day to focused study. Even short daily sessions compound into much better results over a term."),
        "faculty": ("Study skills", "Recommend the student for after-school study programmes or homework clubs.")
    },
    "Wants_Higher_Education": {
        "student": ("Goals", "Speaking with a career counsellor about pathways that match your interests can make school feel more purposeful."),
        "faculty": ("Motivation", "Arrange a one-on-one career guidance session to connect education to real-world goals.")
    },
    "Number_of_Failures": {
        "student": ("Academic recovery", "Past failures do not define your future. Ask your teacher for a targeted catch-up plan before the next assessment."),
        "faculty": ("Academic intervention", "Arrange targeted support sessions and a structured recovery plan with clear short-term milestones.")
    },
    "Number_of_Absences": {
        "student": ("Attendance", "Try to identify what is making it hard to attend and speak with your counsellor — transport, health, or personal issues can all be addressed."),
        "faculty": ("Attendance monitoring", "Flag for the attendance monitoring programme. Contact parents after 3 consecutive unexplained absences.")
    },
    "Social_Risk": {
        "student": ("Lifestyle", "Try to keep weekday commitments light so your energy is available for school the next day."),
        "faculty": ("Wellbeing", "Refer to the school welfare team for a confidential wellbeing check.")
    },
    "Family_Support": {
        "student": ("Home support", "If home feels unsupportive, speak with a trusted teacher or counsellor — they can provide practical help."),
        "faculty": ("Family engagement", "Attempt a positive check-in call with parents before raising concerns. Involve welfare officer if needed.")
    },
    "Parental_Engagement": {
        "student": ("Home environment", "Try sharing your school progress with a family member, even briefly — it can build more support around you."),
        "faculty": ("Parental outreach", "Send a positive progress update home to open communication channels with the family.")
    },
    "Internet_Access": {
        "student": ("Resources", "Ask your school about device loan programmes or using the library for online homework."),
        "faculty": ("Digital inclusion", "Flag for the school device loan programme and ensure homework has offline-friendly alternatives.")
    },
    "Travel_Time": {
        "student": ("Commute", "Try using travel time productively — reviewing notes can turn a long commute into study time."),
        "faculty": ("Logistics", "Explore whether transport support or flexible scheduling is available for this student.")
    },
    "Health_Status": {
        "student": ("Health", "Speak with the school nurse or your doctor — additional support or accommodations may be available to you."),
        "faculty": ("Health support", "Refer to the school nurse and consider flexible deadlines if the student has ongoing health difficulties.")
    },
    "Family_Relationship": {
        "student": ("Relationships", "Your school counsellor offers a confidential space to talk through difficult family situations."),
        "faculty": ("Family dynamics", "Handle sensitively and consider involving the welfare officer if family conflict is affecting attendance.")
    }
}

def preprocess(data):
    row = pd.DataFrame([data])
    for col, le in encoders.items():
        if col in row.columns:
            try:
                row[col] = le.transform(row[col].astype(str))
            except ValueError:
                row[col] = 0
    row["Social_Risk"]         = row["Going_Out"] + row["Weekend_Alcohol_Consumption"] + row["Weekday_Alcohol_Consumption"]
    row["Parental_Engagement"] = row["Mother_Education"] + row["Father_Education"] + row["Family_Relationship"]
    row["Support_Deficit"]     = ((row["School_Support"] == 0) & (row["Family_Support"] == 0)).astype(int)
    return scaler.transform(row[feat_cols])

def get_shap_factors(row_scaled):
    import shap
    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(row_scaled)[0]
    shap_series = pd.Series(shap_vals, index=feat_cols)
    return shap_series.nlargest(4).index.tolist(), shap_series.nsmallest(3).index.tolist()

def build_suggestions(top_risk, probability):
    student_sugs, faculty_sugs, seen = [], [], set()

    for factor in top_risk:
        if factor in SUGGESTIONS and factor not in seen:
            s = SUGGESTIONS[factor]["student"]
            f = SUGGESTIONS[factor]["faculty"]
            student_sugs.append({"category": s[0], "action": s[1]})
            faculty_sugs.append({"category": f[0], "action": f[1]})
            seen.add(factor)
        if len(student_sugs) >= 3:
            break

    for factor, sug in SUGGESTIONS.items():
        if len(student_sugs) >= 3:
            break
        if factor not in seen:
            student_sugs.append({"category": sug["student"][0], "action": sug["student"][1]})
            faculty_sugs.append({"category": sug["faculty"][0], "action": sug["faculty"][1]})
            seen.add(factor)

    risk_summary = (
        f"This student has a {round(probability*100)}% probability of dropping out. "
        f"The primary concerns identified are: {', '.join(top_risk[:3])}. "
        f"Early and empathetic intervention focusing on these areas is recommended."
    )
    return risk_summary, student_sugs[:3], faculty_sugs[:3]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data       = request.json
        row_scaled = preprocess(data)
        proba      = model.predict_proba(row_scaled)[0][1]
        label      = "Dropout Risk" if proba >= threshold else "Low Risk"
        top_risk, top_protect = get_shap_factors(row_scaled)
        risk_summary, student_sugs, faculty_sugs = build_suggestions(top_risk, proba)

        return jsonify({
            "probability":            round(float(proba), 3),
            "label":                  label,
            "threshold":              round(float(threshold), 3),
            "top_risk_factors":       top_risk,
            "top_protective_factors": top_protect,
            "risk_summary":           risk_summary,
            "student_suggestions":    student_sugs,
            "faculty_suggestions":    faculty_sugs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "threshold": round(float(threshold), 3)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
