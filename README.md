# Hotel Booking Cancellation Prediction

This project provides a machine learning solution to predict hotel booking cancellations using a **Random Forest** model.  
It includes:

- A **Flask** web application for uploading CSV files, preprocessing data, and generating predictions.
- A **Jupyter Notebook** for data analysis, feature engineering, and model training/evaluation.

---

## 📁 Project Structure

- `app.py`: Flask app for file upload, preprocessing, and prediction.
- `Hassan Obaia Task 3 Intern 1.ipynb`: Jupyter notebook with full ML pipeline and model comparisons.
- `random_forest_model.pkl`: Pre-trained model used in the Flask app.
- `uploads/`: Folder to store uploaded files and prediction results.
- `templates/index.html`: HTML upload page for the Flask app.

---

## ⚙️ Prerequisites

- Python 3.8+
- Required packages:
  ```bash
  pip install flask pandas numpy scikit-learn category_encoders joblib xgboost


* Jupyter Notebook (for `.ipynb` analysis)
* `random_forest_model.pkl` (required for prediction)

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*Alternatively, install packages manually.*

### 3. Prepare the Model

* Place `random_forest_model.pkl` in the project root.
* To retrain the model, run the Jupyter notebook:

```bash
jupyter notebook "Hassan Obaia Task 3 Intern 1.ipynb"
```

### 4. Run the Flask Application

```bash
python app.py
```

App will be available at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📦 Usage

### Web App:

* Go to `http://127.0.0.1:5000`
* Upload a hotel booking CSV file (see expected format below)
* Receive a `predictions.csv` file with cancellation predictions (1 = Canceled, 0 = Not Canceled)

### Expected Input Format:

Your CSV should include the following columns:

```
Booking_ID, number of adults, number of children, number of weekend nights, number of week nights,
type of meal, car parking space, room type, lead time, market segment type, repeated,
P-C, P-not-C, average price, special requests, date of reservation, booking status
```

---

## 🔄 Data Preprocessing (`app.py`)

* Cleans column names and formats dates
* Extracts features like: `year`, `month`, `is_weekend`, `total_nights`, `guest_type`
* Applies log transformation on skewed columns
* Drops unnecessary columns (e.g., Booking\_ID, P-C, P-not-C)
* Encodes categorical features (LabelEncoder, TargetEncoder)
* Uses dummy variables for room type and meal
* Scales numerical values using `RobustScaler`

---

## 🧠 Model Training (`.ipynb`)

### Evaluated Models:

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 78%      |
| KNN                 | 84%      |
| Decision Tree       | 87%      |
| Random Forest       | 89%      |
| XGBoost             | 89%      |

* Hyperparameter tuning via `GridSearchCV` and `RandomizedSearchCV`
* **Random Forest** selected for deployment due to highest stable performance.

---

## 📝 Notes

* Ensure input CSV matches the required format.
* `random_forest_model.pkl` must be in root directory for the Flask app.
* `Hassan Obaia Task 3 Intern 1.ipynb` requires dataset (`first inten project.csv`) for retraining.
* Add error handling and security before deploying publicly.
