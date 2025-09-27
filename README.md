# Customer Churn Prediction (Google Cloud + Scikit-learn)

This project builds a **machine learning pipeline** to predict customer churn in a telco dataset.
It uses **Google BigQuery** as the data source (cloud) and **scikit-learn** for modeling. A small
CSV sample is included to make the project reproducible without cloud access.

---

## 📊 Dataset

- **Primary source (cloud):** a table in Google BigQuery (replace with your path):
  ```text
  YOUR_PROJECT_ID.customer_churn.telco_churn
  ```
  > In your case: `churn-analytics-473407.customer_churn.telco_churn` (private).

- **Local sample (reproducible):** `data/sample_churn.csv` (e.g., first ~200 rows).  
  Use this if you don’t have BigQuery access.

Typical columns include:
`customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn`.

---

## ⚙️ Setup

```bash
# (Recommended) Create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, the core libs are:
```bash
pip install pandas scikit-learn matplotlib google-cloud-bigquery python-dotenv
```

---

## 🚀 How to Run

### Option A — Run locally with the sample CSV (no cloud needed)
Edit the notebook or `src/pipeline.py` to load the local file:
```python
import pandas as pd
df = pd.read_csv("data/sample_churn.csv")
```

### Option B — Run against BigQuery (Google Cloud)
**Colab (interactive) auth:**
```python
from google.colab import auth
auth.authenticate_user()

from google.cloud import bigquery
client = bigquery.Client(project="YOUR_PROJECT_ID")
df = client.query("SELECT * FROM `YOUR_PROJECT_ID.customer_churn.telco_churn`").to_dataframe()
```

**Local (service account) auth:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"  # Windows: setx GOOGLE_APPLICATION_CREDENTIALS "C:\path\credentials.json"
```
Then in Python, the BigQuery client will pick it up automatically:
```python
from google.cloud import bigquery
client = bigquery.Client(project="YOUR_PROJECT_ID")
```

---

## 🧠 Modeling Approach

The project uses a **scikit-learn Pipeline** that:
1) Converts `TotalCharges` to numeric and imputes missing values
2) One-hot encodes categorical features
3) Passes numeric & boolean features through
4) Trains a classifier

Models compared:
- **Logistic Regression** (`class_weight="balanced"`)
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

Train/test split uses `stratify=y`, `test_size=0.2`, and `random_state=42`.

---

## 📈 Results (from our runs on the sample telco churn data)

| Model                          | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|--------------------------------|----------|-------------------|----------------|------------|
| Logistic Regression (balanced) | **0.74** | 0.50              | **0.76**       | **0.61**   |
| Random Forest (threshold 0.4)  | 0.64     | 0.42              | **0.89**       | 0.57       |
| Gradient Boosting              | **0.79** | **0.63**          | 0.53           | 0.57       |

> Notes:
> - Lower thresholds increase **recall** (catch more churners) at the cost of **precision** (more false positives).
> - `class_weight="balanced"` helps the minority class (Churn=1) for Logistic Regression.

---

## 🔍 Feature Importance

**Tree-based models (RF/GB):**
```python
# Using a fitted Gradient Boosting pipeline named `clf_gb`
import numpy as np, pandas as pd

pre = clf_gb.named_steps["preprocessor"]
feature_names = pre.get_feature_names_out()
importances = clf_gb.named_steps["model"].feature_importances_

imp = pd.DataFrame({"feature": feature_names, "importance": importances})\
        .sort_values("importance", ascending=False)
print(imp.head(15))
```

**What we typically see as most influential:**
- **tenure** (months with the company)
- **Contract** (Month-to-month drives churn up)
- **MonthlyCharges**
- **OnlineSecurity / TechSupport / InternetService**

---

## 📂 Project Structure

```
/churn-prediction/
│── notebooks/
│    └── churn_analysis.ipynb     # Jupyter/Colab notebook
│── src/
│    └── pipeline.py              # Reproducible sklearn pipeline
│── data/
│    └── sample_churn.csv         # Reduced dataset for local runs
│── requirements.txt
│── README.md
│── .gitignore
```

---

## 🔒 Security & Best Practices

- **Do NOT commit credentials** or tokens.
- Add a `.gitignore`:
  ```gitignore
  .env
  *.json
  .venv/
  __pycache__/
  .ipynb_checkpoints/
  data/*.csv
  !data/sample_churn.csv
  ```
- Prefer environment variables (e.g., `GOOGLE_APPLICATION_CREDENTIALS`).

---

## ✅ Next Steps

- Hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`)
- Try **XGBoost / LightGBM / CatBoost**
- Deploy a prediction API (FastAPI/Flask) on **Cloud Run**
- Connect predictions & metrics to **Looker Studio** dashboards
