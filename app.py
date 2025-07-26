from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, RobustScaler
import category_encoders as ce  

model = joblib.load("random_forest_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return "No file uploaded!"

        file = request.files["file"]
        if file.filename == "":
            return "Please select a file!"

        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        df = pd.read_csv(file_path)

        
        df.columns = df.columns.str.strip()

        df['date of reservation'] = pd.to_datetime(df['date of reservation'], errors='coerce')
        df.dropna(inplace=True)

        df["year"] = df["date of reservation"].dt.year.astype(int)
        df["month"] = df["date of reservation"].dt.month.astype(int)

        df["is_weekend"] = df["date of reservation"].dt.weekday.isin([5, 6]).astype(int)

        df['total nights'] = df['number of weekend nights'] + df['number of week nights']

        df['guest type'] = df.apply(lambda x: 'Family' if x['number of children'] > 0 else 'Individual/Couple', axis=1)

        df.drop(columns=["date of reservation", 'P-C', 'P-not-C',
                         'Booking_ID', 'number of adults', 'number of children',
                         'number of weekend nights', 'number of week nights'],
                inplace=True)

        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        skewness = df[numerical_cols].skew()
        skewed_cols = skewness[abs(skewness) > 1].index
        df[skewed_cols] = np.log1p(df[skewed_cols])

        df["booking status"] = LabelEncoder().fit_transform(df["booking status"])
        df["guest type"] = LabelEncoder().fit_transform(df["guest type"])

        target_encoder = ce.TargetEncoder(cols=["market segment type"])
        df["market segment type_encoded"] = target_encoder.fit_transform(df["market segment type"], df["booking status"])
        df.drop(columns=["market segment type"], inplace=True)

        df = pd.get_dummies(df, columns=['room type', 'type of meal'])

        columns_to_scale = ['lead time', 'average price', 'total nights']
        scaler = RobustScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

        df_features = df.drop(columns=["booking status"], errors="ignore")
        df["Prediction"] = model.predict(df_features)

        result_path = os.path.join("uploads", "predictions.csv")
        df.to_csv(result_path, index=False)

        return f"Predictions generated! <a href='/download'>Download Results</a>"

    except Exception as e:
        return str(e)

@app.route("/download")
def download():
    return send_file("uploads/predictions.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
