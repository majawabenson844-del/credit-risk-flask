from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ===============================
# Load Models & Artifacts
# ===============================
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

model_svm = safe_load("models/svm_model.pkl")
model_rf = safe_load("models/rf_model.pkl")
model_ensemble = safe_load("models/ensemble_model.pkl")
scaler = safe_load("models/scaler.pkl")
encoder = safe_load("models/encoder.pkl")

# Load dataset for dropdown options
try:
    data = pd.read_csv("combined_solar_dataset.csv")
except Exception as e:
    print(f"Error loading dataset: {e}")
    data = pd.DataFrame()

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return render_template("home.html", title="Home")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            age = int(request.form.get("Age"))
            loan_amount = int(request.form.get("Loan_amt"))
            gender = request.form.get("Gender")
            marital = request.form.get("Marital_Status")
            employment = request.form.get("Employment_Status")
            residence = request.form.get("Residence_Area")
            home = request.form.get("Home_Ownership")
            dependents = int(request.form.get("Number_of_Dependants"))
        except Exception:
            return render_template("predict.html", title="Predict",
                                   error="⚠️ Invalid input provided.",
                                   gender_options=data["Gender"].unique().tolist(),
                                   marital_options=data["Marital_Status"].unique().tolist(),
                                   employment_options=data["Employment_Status"].unique().tolist(),
                                   residence_options=data["Residence_Area"].unique().tolist(),
                                   home_options=data["Home_Ownership"].unique().tolist())

        # Validation
        if age < 18:
            return render_template("predict.html", title="Predict",
                                   error="⚠️ Age must be at least 18 years.",
                                   gender_options=data["Gender"].unique().tolist(),
                                   marital_options=data["Marital_Status"].unique().tolist(),
                                   employment_options=data["Employment_Status"].unique().tolist(),
                                   residence_options=data["Residence_Area"].unique().tolist(),
                                   home_options=data["Home_Ownership"].unique().tolist())
        if loan_amount < 10:
            return render_template("predict.html", title="Predict",
                                   error="⚠️ Loan amount must be at least 10.",
                                   gender_options=data["Gender"].unique().tolist(),
                                   marital_options=data["Marital_Status"].unique().tolist(),
                                   employment_options=data["Employment_Status"].unique().tolist(),
                                   residence_options=data["Residence_Area"].unique().tolist(),
                                   home_options=data["Home_Ownership"].unique().tolist())

        # Build dataframe with dataset column names
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Marital_Status": marital,
            "Employment_Status": employment,
            "Residence_Area": residence,
            "Home_Ownership": home,
            "Number_of_Dependants": dependents,
            "Loan amt": loan_amount
        }])

        # 🔧 Rename to match training feature names
        input_df = input_df.rename(columns={
            "Employment_Status": "Employment",
            "Residence_Area": "Residence",
            "Number_of_Dependants": "Number_Dependents",
            "Loan amt": "Loan_Amount"
        })

        # Encode + scale
        categorical_cols = ['Gender','Marital_Status','Employment','Residence','Home_Ownership']
        continuous_cols = ['Age','Number_Dependents','Loan_Amount']

        encoded_cat = encoder.transform(input_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_cat, columns=categorical_cols)
        cont_df = input_df[continuous_cols].astype(float)
        final_df = pd.concat([encoded_df, cont_df], axis=1)

        scaled = scaler.transform(final_df)

        # 🔀 Predict across all models
        results = {}
        approvals = 0
        rejections = 0

        if model_svm:
            pred = model_svm.predict(scaled)[0]
            probs = model_svm.predict_proba(scaled)[0]
            results["SVM"] = {"prediction": pred, "probs": probs}
            approvals += (pred == 1)
            rejections += (pred == 0)

        if model_rf:
            pred = model_rf.predict(scaled)[0]
            probs = model_rf.predict_proba(scaled)[0]
            results["Random Forest"] = {"prediction": pred, "probs": probs}
            approvals += (pred == 1)
            rejections += (pred == 0)

        if model_ensemble:
            pred = model_ensemble.predict(scaled)[0]
            probs = model_ensemble.predict_proba(scaled)[0]
            results["Ensemble"] = {"prediction": pred, "probs": probs}
            approvals += (pred == 1)
            rejections += (pred == 0)

        # Build summary
        summary = f"{approvals} model(s) approved, {rejections} model(s) rejected."

        return render_template("predict.html", title="Predict",
                               results=results, summary=summary,
                               gender_options=data["Gender"].unique().tolist(),
                               marital_options=data["Marital_Status"].unique().tolist(),
                               employment_options=data["Employment_Status"].unique().tolist(),
                               residence_options=data["Residence_Area"].unique().tolist(),
                               home_options=data["Home_Ownership"].unique().tolist())

    # GET request → show form with dropdowns
    return render_template("predict.html", title="Predict",
                           gender_options=data["Gender"].unique().tolist(),
                           marital_options=data["Marital_Status"].unique().tolist(),
                           employment_options=data["Employment_Status"].unique().tolist(),
                           residence_options=data["Residence_Area"].unique().tolist(),
                           home_options=data["Home_Ownership"].unique().tolist())

@app.route("/model-info")
def model_info():
    return render_template("model_info.html", title="Model Info")

@app.route("/feature-guide")
def feature_guide():
    return render_template("feature_guide.html", title="Feature Guide", data=data)

@app.route("/about")
def about():
    return render_template("about.html", title="About")

if __name__ == "__main__":
    app.run(debug=True, port=8080)

