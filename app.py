from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load Model
model = joblib.load("models/model.pkl")


# ==========================
# HOME PAGE
# ==========================
@app.route("/")
def home():
    return render_template("home.html")


# ==========================
# DASHBOARD
# ==========================
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ==========================
# PREDICTION PAGE
# ==========================
@app.route("/predict")
def predict():
    return render_template("predict.html")


# ==========================
# RESULT PAGE
# ==========================
@app.route("/result", methods=["POST"])
def result():

    OverallQual = float(request.form["OverallQual"])
    GrLivArea = float(request.form["GrLivArea"])
    SecondFlrSF = float(request.form["SecondFlrSF"])
    TotalBsmtSF = float(request.form["TotalBsmtSF"])
    BsmtFinSF1 = float(request.form["BsmtFinSF1"])
    FirstFlrSF = float(request.form["FirstFlrSF"])
    LotArea = float(request.form["LotArea"])
    GarageArea = float(request.form["GarageArea"])
    GarageCars = float(request.form["GarageCars"])
    YearBuilt = float(request.form["YearBuilt"])

    features = np.array([[
        OverallQual,
        GrLivArea,
        SecondFlrSF,
        TotalBsmtSF,
        BsmtFinSF1,
        FirstFlrSF,
        LotArea,
        GarageArea,
        GarageCars,
        YearBuilt
    ]])

    prediction = model.predict(features)[0]

    return render_template(
        "result.html",
        prediction=prediction
    )


# ==========================
# ANALYTICS PAGE
# ==========================
@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


# ==========================
# AI INSIGHTS PAGE
# ==========================
@app.route("/insights")
def insights():
    return render_template("insights.html")


if __name__ == "__main__":
    app.run(debug=True)