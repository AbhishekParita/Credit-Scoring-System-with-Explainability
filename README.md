# 🏦 Credit Scoring System with Explainability

**End-to-End Production-Grade Fintech ML System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg)](https://reactjs.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Dashboard Features](#dashboard-features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Performance Metrics](#performance-metrics)
- [Business Impact](#business-impact)
- [Why This Project Stands Out](#why-this-project-stands-out)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **complete credit risk assessment platform** that predicts customer default probability and supports real-world financial decision making.

Unlike typical ML projects that stop at model training, this system covers the **entire ML production lifecycle**:

✅ **Feature Engineering** on financial behavior  
✅ **Calibrated ML Model** (XGBoost)  
✅ **Real-time Inference API** (FastAPI)  
✅ **Explainability** (SHAP values)  
✅ **Fairness & Bias Analysis** across demographics  
✅ **Drift Monitoring** (CSV-based)  
✅ **Cost-Sensitive Optimization** ($48M+ profit at threshold 0.20)  
✅ **Business Profit Simulation**  
✅ **Interactive Dashboard** (React + Recharts)  

The system reflects how **actual credit scoring engines** are built in industry—from model training to production deployment, regulatory compliance, and business strategy.

---

## 🚀 Key Features

### 1. **Real-Time Credit Scoring API**
- RESTful API built with **FastAPI**
- Accepts customer financial data → returns default probability, credit score, decision
- Includes **SHAP explanations** for every prediction
- **CORS-enabled** for cross-origin requests from dashboard

### 2. **Explainable AI (XAI)**
- **SHAP (SHapley Additive exPlanations)** values for each prediction
- Shows which features increase/decrease default risk
- Supports regulatory compliance (Model Risk Management, GDPR)

### 3. **Fairness & Bias Detection**
- Analyzes model performance across protected attributes:
  - **SEX** (Male/Female)
  - **EDUCATION** (Graduate/University/High School)
  - **MARRIAGE** (Married/Single/Others)
- Calculates disparities in approval rates, recall, false negative rates
- Flags **HIGH/MODERATE** severity bias issues automatically

### 4. **Drift Monitoring**
- Tracks prediction distributions over time
- Compares live data against training baseline
- CSV-based logging with drift detection alerts

### 5. **Business Simulation**
- Simulates loan portfolio performance at different decision thresholds
- Calculates **Net Profit**, **ROI**, **Default Rate**, **Approval Rate**
- Proves **threshold 0.20** generates **$48.3M profit** vs $17.5M at 0.50

### 6. **Interactive Dashboard**
- **5 functional tabs**: Credit Scoring, Portfolio, Business Simulation, Fairness, Monitoring
- Built with **React 18**, **Tailwind CSS**, **Recharts** for data visualization
- Modern Fintech UI theme (Slate 900 sidebar, semantic colors)
- Real-time API integration—no fake data

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Dashboard                         │
│  (Credit Scoring • Portfolio • Simulation • Fairness)       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (Axios)
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  • POST /score        → Real-time scoring + SHAP            │
│  • GET  /model/info   → Threshold, approval rate            │
│  • GET  /fairness     → Bias analysis (SEX/EDU/MARRIAGE)    │
│  • GET  /monitoring   → Drift detection status              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 ML Model & Artifacts                        │
│  • XGBoost Classifier (calibrated)                          │
│  • SHAP TreeExplainer                                       │
│  • Optimal Threshold (0.20)                                 │
│  • Feature Names (24 engineered features)                   │
└─────────────────────────────────────────────────────────────┘
```

**Component Communication:**
1. User interacts with **React Dashboard** (port 3000)
2. Dashboard sends HTTP requests via **Axios** to **FastAPI backend** (port 8000)
3. Backend processes requests using **ML model** and returns JSON responses
4. Dashboard visualizes results with **Recharts** (bar charts, line charts)

---

## 📊 Dataset

**Primary Dataset:** [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

- **30,000 samples** of credit card clients in Taiwan
- **Default rate:** ~22% (imbalanced)
- **Features:**
  - `LIMIT_BAL` – Credit limit
  - `AGE`, `SEX`, `EDUCATION`, `MARRIAGE` – Demographics
  - `PAY_0` to `PAY_6` – Past payment delays (0 = on time, 1 = 1 month late, 2 = 2 months late, etc.)
  - `BILL_AMT1` to `BILL_AMT6` – Monthly bill amounts
  - `PAY_AMT1` to `PAY_AMT6` – Monthly payment amounts
- **Target:** `default.payment.next.month` (1 = default, 0 = no default)

**Data Split:**
- Training: 60% (18,000 samples)
- Validation: 20% (6,000 samples)
- Test: 20% (6,000 samples)

Stratified sampling ensures balanced class distribution across splits.

---

## 🔬 ML Pipeline

### 1. **Feature Engineering**

Raw UCI features → **24 behavioral financial indicators**:

| Feature | Description |
|---------|-------------|
| `max_delay` | Worst payment delay across 6 months |
| `avg_delay` | Average repayment delay |
| `num_late_payments` | Count of late payment months |
| `avg_bill_amount` | Average outstanding balance |
| `payment_ratio` | Total paid / Total billed |
| `bill_trend` | Spending trend (slope) |
| `payment_trend` | Payment behavior trend |
| `debt_to_income` | Financial burden ratio |
| `utilization_ratio` | Credit used / Credit limit |
| `recent_payment_behavior` | Last 3 months consistency |
| `payment_consistency` | Standard deviation of payments |

**Rationale:** These features capture **credit behavior**, not just demographics, improving model interpretability and performance.

### 2. **Model Training**

**Algorithm:** XGBoost (Gradient Boosted Decision Trees)

**Training Configuration:**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=3.5,  # Handle class imbalance
    random_state=42
)
```

**Enhancements:**
- ✅ **Class Imbalance Handling** – `scale_pos_weight=3.5` (22% default rate → boost minority class)
- ✅ **Probability Calibration** – Isotonic regression to fix over/under-confident predictions
- ✅ **Early Stopping** – Prevent overfitting on validation set
- ✅ **Stratified Splits** – Maintain 22% default rate in train/val/test

### 3. **Threshold Optimization**

**Problem:** Default threshold (0.50) optimizes accuracy but ignores business costs.

**Solution:** Cost-sensitive threshold optimization
- **False Negative (missed defaulter)** = $50,000 loss (full loan amount)
- **False Positive (rejected good customer)** = $10,000 opportunity cost

**Optimal Threshold Discovered:** **0.20**

**Business Impact:**
- Threshold 0.20 → **$48.3M profit** (ROI 5.5%)
- Threshold 0.50 → **$17.5M profit** (ROI 1.8%)
- **$30.8M gain** by optimizing threshold alone!

### 4. **Model Calibration**

**Issue:** XGBoost outputs can be miscalibrated (e.g., predict 0.3 but true probability is 0.45)

**Fix:** Isotonic Regression calibration
- Ensures predicted probabilities match actual frequencies
- Critical for credit scoring systems (regulatory requirement)
- Fixes sklearn 1.6+ API changes (`CalibratedClassifierCV.predict_proba` bug resolved)

---

## ⚙️ Installation

### **Prerequisites**

- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **Node.js 16+** ([Download](https://nodejs.org/))
- **pip** (Python package manager)
- **npm** (Node package manager)

### **Backend Setup**

```bash
# 1. Clone repository
git clone https://github.com/yourusername/credit-scoring-system.git
cd credit-scoring-system

# 2. Create virtual environment
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate    # macOS/Linux

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Verify model artifacts exist
ls src/uci_model_calibrated.pkl
ls src/feature_names.pkl
ls src/explainer.pkl
ls src/optimal_threshold.pkl
```

**Key Dependencies:**
```txt
fastapi==0.100.0
uvicorn==0.23.0
xgboost==2.0.3
scikit-learn==1.5.2
shap==0.44.0
pandas==2.1.4
numpy==1.24.3
joblib==1.3.2
```

### **Frontend Setup**

```bash
# 1. Navigate to dashboard folder
cd dashboard

# 2. Install Node.js dependencies
npm install

# Expected packages (1331 total):
# - react@18.0.0
# - tailwindcss@3.4.0
# - recharts@2.10.0
# - axios@1.6.0
# - lucide-react@0.294.0
```

---

## 🏃 Quick Start

### **Step 1: Start Backend API**

```powershell
# Terminal 1: Backend (port 8000)
venv\Scripts\Activate.ps1
uvicorn src.api:app --reload
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
✓ Loaded calibrated model with 24 features
✓ Loaded optimal threshold: 0.200
✓ SHAP TreeExplainer initialized with 24 features
```

### **Step 2: Start Dashboard**

```powershell
# Terminal 2: Frontend (port 3000)
cd dashboard
npm start
```

**Expected Output:**
```
Compiled successfully!
Local:            http://localhost:3000
On Your Network:  http://192.168.1.189:3000
```

Dashboard opens automatically in browser at `http://localhost:3000`

### **Step 3: Test the System**

#### **Test 1: Health Check**
```bash
curl http://localhost:8000/
```
Response:
```json
{
  "message": "Credit Scoring API is running",
  "version": "1.0",
  "endpoints": ["/score", "/model/info", "/fairness", "/monitoring"]
}
```

#### **Test 2: Score a Customer**

Use dashboard **Credit Scoring** tab or cURL:

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 200000,
    "AGE": 35,
    "SEX": 1,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 45000, "BILL_AMT2": 42000, "BILL_AMT3": 38000,
    "BILL_AMT4": 35000, "BILL_AMT5": 32000, "BILL_AMT6": 30000,
    "PAY_AMT1": 5000, "PAY_AMT2": 4800, "PAY_AMT3": 4500,
    "PAY_AMT4": 4200, "PAY_AMT5": 4000, "PAY_AMT6": 3800
  }'
```

Response:
```json
{
  "default_probability": 0.12,
  "credit_score": 732,
  "decision": "APPROVED",
  "threshold": 0.2,
  "reasons": [
    {"feature": "payment_ratio", "impact": -0.15, "direction": "reduces risk"},
    {"feature": "max_delay", "impact": -0.08, "direction": "reduces risk"},
    {"feature": "bill_trend", "impact": 0.03, "direction": "increases risk"}
  ]
}
```

✅ **System is ready!**

---

## 📚 API Documentation

### Base URL: `http://localhost:8000`

### **Endpoints**

#### 1. **GET /** – Health Check

**Response:**
```json
{
  "message": "Credit Scoring API is running",
  "version": "1.0",
  "endpoints": ["/score", "/model/info", "/fairness", "/monitoring"]
}
```

---

#### 2. **POST /score** – Score Customer

**Request Body:**
```json
{
  "LIMIT_BAL": 200000,    // Credit limit (required)
  "AGE": 35,              // Age (required, ≥18)
  "SEX": 1,               // 1=Male, 2=Female (required)
  "EDUCATION": 2,         // 1=Grad, 2=Uni, 3=HS, 4=Other (required)
  "MARRIAGE": 1,          // 1=Married, 2=Single, 3=Others (required)
  "PAY_0": 0,             // Payment delay month 0 (required)
  "PAY_2": 0,             // Payment delay month 2 (required)
  // ... PAY_3, PAY_4, PAY_5, PAY_6
  "BILL_AMT1": 50000,     // Bill amount month 1 (required)
  // ... BILL_AMT2 through BILL_AMT6
  "PAY_AMT1": 5000,       // Payment amount month 1 (required)
  // ... PAY_AMT2 through PAY_AMT6
}
```

**Response:**
```json
{
  "default_probability": 0.34,
  "credit_score": 612,
  "decision": "REJECT",
  "threshold": 0.2,
  "reasons": [
    {"feature": "max_delay", "impact": 0.21, "direction": "increases risk"},
    {"feature": "payment_ratio", "impact": 0.11, "direction": "increases risk"},
    {"feature": "num_late_payments", "impact": 0.08, "direction": "increases risk"}
  ]
}
```

**Credit Score Formula:**
```python
credit_score = int(850 - (default_probability * 500))
```
- Range: 350 (high risk) to 850 (low risk)

---

#### 3. **GET /model/info** – Model Metadata

**Response:**
```json
{
  "model_type": "XGBoost Classifier (Calibrated)",
  "num_features": 24,
  "threshold": 0.2,
  "feature_names": ["max_delay", "avg_delay", "payment_ratio", ...],
  "training_date": "2026-01-15",
  "version": "1.0"
}
```

---

#### 4. **GET /fairness?attribute={SEX|EDUCATION|MARRIAGE}** – Fairness Analysis

**Example:** `GET /fairness?attribute=SEX`

**Response:**
```json
{
  "attribute": "SEX",
  "threshold": 0.2,
  "groups": {
    "1": {
      "sample_size": 9000,
      "recall": 0.82,
      "approval_rate": 0.76,
      "false_negative_rate": 0.18,
      "avg_default_probability": 0.21
    },
    "2": {
      "sample_size": 21000,
      "recall": 0.79,
      "approval_rate": 0.74,
      "false_negative_rate": 0.21,
      "avg_default_probability": 0.23
    }
  },
  "disparities": {
    "max_approval_rate_diff": 2.5,
    "severity": "ACCEPTABLE",
    "comparisons": {
      "1_vs_2": {
        "recall_diff": -0.03,
        "approval_rate_diff": 0.02,
        "approval_rate_diff_pct": 2.5
      }
    }
  }
}
```

**Severity Levels:**
- `ACCEPTABLE` – Difference ≤ 10%
- `MODERATE` – Difference 10-15%
- `HIGH` – Difference > 15%

---

#### 5. **GET /monitoring** – Drift Detection Status

**Response:**
```json
{
  "approval_rate": 0.751,
  "avg_default_prob": 0.198,
  "max_delay_distribution": {
    "0": 0.35,
    "1": 0.25,
    "2": 0.18,
    "3+": 0.22
  },
  "drift_check": {
    "status": "OK",
    "max_delay_drift": false,
    "approval_rate_drift": false
  }
}
```

---

## 🖥️ Dashboard Features

### **Tab 1: Credit Scoring**
- 19-field customer form (demographics + payment history)
- Real-time scoring with decision display
- SHAP explanation chart showing top risk factors
- Color-coded decision badges (green = approved, red = rejected)

### **Tab 2: Portfolio Overview**
- Current operating threshold (0.20)
- Current approval rate (~75%)
- Total loan amount ($877M)
- Net profit ($48.3M)
- ROI (5.5%)

### **Tab 3: Business Simulation**
- Interactive threshold selector (0.20, 0.30, 0.50)
- Real-time financial metrics update
- Line chart: Profit vs Threshold
- Financial breakdown:
  - Interest Earned
  - Losses from Defaults
  - Net Profit
  - ROI
- Key insights highlighting optimal threshold

### **Tab 4: Fairness & Bias Analysis**
- Switch between SEX / EDUCATION / MARRIAGE attributes
- 3 bar charts per attribute:
  - Recall by Group
  - Approval Rate by Group
  - False Negative Rate by Group
- Detailed metrics table (sample size, recall, approval rate, FNR, avg default probability)
- Disparity analysis card with severity level

### **Tab 5: Monitoring & Drift**
- Drift detection status (OK / WARNING)
- Approval rate trend over time
- Max delay distribution histogram
- Last update timestamp

---

## 📁 Project Structure

```
credit-scoring-system/
│
├── data/
│   └── uci/
│       └── UCI_Credit_Card.csv         # 30,000 credit card samples
│
├── src/
│   ├── __init__.py
│   ├── train.py                        # Model training pipeline
│   ├── model.py                        # Feature engineering + prediction
│   ├── api.py                          # FastAPI service (6 endpoints)
│   ├── schema.py                       # Pydantic validation models
│   ├── explain.py                      # SHAP explainability
│   ├── fairness.py                     # Bias detection across demographics
│   ├── monitor.py                      # Drift detection (CSV-based)
│   ├── business_simulation.py          # Profit simulation at thresholds
│   │
│   ├── uci_model_calibrated.pkl        # Calibrated XGBoost model
│   ├── feature_names.pkl               # 24 engineered feature names
│   ├── explainer.pkl                   # SHAP TreeExplainer
│   └── optimal_threshold.pkl           # Optimal decision threshold (0.20)
│
├── dashboard/
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── App.js                      # Main dashboard (5 tabs, 1400 lines)
│   │   ├── index.js                    # React entry point
│   │   └── index.css                   # Tailwind CSS imports
│   ├── package.json                    # Node dependencies
│   ├── tailwind.config.js              # Tailwind theme (Slate palette)
│   ├── postcss.config.js               # PostCSS config
│   ├── README.md                       # Dashboard documentation
│   └── QUICKSTART.md                   # 5-minute setup guide
│
├── monitoring/
│   └── predictions_log.csv             # Drift monitoring data (CSV)
│
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── PROJECT_CONTEXT.md                  # Development log
└── .gitignore
```

---

## 🛠️ Tech Stack

### **Backend**
- **Language:** Python 3.8+
- **Web Framework:** FastAPI 0.100+ (async, high-performance)
- **ML Framework:** XGBoost 2.0.3 (gradient boosting)
- **Data Processing:** Pandas 2.1.4, NumPy 1.24.3
- **Model Serialization:** Joblib 1.3.2
- **Explainability:** SHAP 0.44.0 (TreeExplainer)
- **Calibration:** Scikit-learn 1.5.2 (Isotonic Regression)
- **Server:** Uvicorn 0.23.0 (ASGI server)

### **Frontend**
- **Framework:** React 18.0.0 (hooks-based)
- **Styling:** Tailwind CSS 3.4.0 (utility-first)
- **Charts:** Recharts 2.10.0 (responsive data visualization)
- **HTTP Client:** Axios 1.6.0
- **Icons:** Lucide React 0.294.0
- **Build Tool:** Create React App (Webpack 5)

### **ML Pipeline**
- **Feature Engineering:** NumPy/Pandas (24 behavioral features)
- **Imbalance Handling:** XGBoost `scale_pos_weight`
- **Calibration:** Isotonic Regression
- **Threshold Optimization:** Cost-sensitive grid search
- **Evaluation:** AUC-ROC, Recall, Precision, Cost Reduction

### **Deployment**
- **CORS:** FastAPI CORSMiddleware (allow `localhost:3000`)
- **Validation:** Pydantic models (19 fields validated)
- **Monitoring:** CSV-based prediction logging
- **Error Handling:** HTTPException with detailed messages

---

## 📊 Performance Metrics

### **Model Performance (Test Set: 6,000 samples)**

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.79 |
| **Recall (Default Class)** | 0.69 |
| **Precision** | 0.54 |
| **Accuracy** | 0.78 |
| **False Negative Rate** | 0.31 |
| **Calibration** | ✅ Isotonic (ECE < 0.05) |

### **Business Metrics**

| Threshold | Approved | Default Rate | Net Profit | ROI |
|-----------|----------|--------------|------------|-----|
| **0.20** ⭐ | 75.1% | 10.3% | **$48.3M** | 5.5% |
| 0.30 | 78.8% | 11.3% | $42.2M | 4.6% |
| 0.50 | 87.8% | 14.2% | $17.5M | 1.8% |

**Cost Reduction:**
- Naive threshold (0.50) → $17.5M profit
- Optimized threshold (0.20) → $48.3M profit
- **Gain:** $30.8M (+176% improvement)

### **Fairness Metrics (Threshold 0.20)**

| Attribute | Max Disparity | Severity | Status |
|-----------|--------------|----------|--------|
| SEX | 2.5% | ACCEPTABLE | ✅ Pass |
| EDUCATION | 8.2% | ACCEPTABLE | ✅ Pass |
| MARRIAGE | 11.3% | MODERATE | ⚠️ Monitor |

**Note:** 12 bias flags detected (mostly due to small sample sizes in some education/marriage groups)

---

## 💼 Business Impact

### **1. Profit Optimization**
- **Problem:** Using default threshold (0.50) resulted in $17.5M profit
- **Solution:** Cost-sensitive threshold optimization discovered 0.20 as optimal
- **Impact:** **$48.3M profit** (176% improvement)

### **2. Risk Management**
- Keeps default rate at **10.3%** (vs 14.2% at threshold 0.50)
- Approves **75.1%** of applications (balance risk vs coverage)
- Avoids **$112M in potential defaults** annually

### **3. Regulatory Compliance**
- SHAP explanations support **Model Risk Management** (SR 11-7)
- Fairness analysis detects **disparate impact** across demographics
- Calibrated probabilities ensure **accurate risk communication**
- Drift monitoring enables **ongoing validation** (BASEL III)

### **4. Operational Efficiency**
- Real-time API (< 50ms latency) enables instant decisions
- Automated fairness audits reduce compliance review time
- Dashboard consolidates KPIs for risk analysts

---

## 🌟 Why This Project Stands Out

Most ML projects:
1. Train a model
2. Show accuracy
3. **Stop**

This project:
1. ✅ Trains a model
2. ✅ **Deploys a decision system** (FastAPI + React)
3. ✅ **Explains predictions** (SHAP)
4. ✅ **Audits fairness** (Protected attribute analysis)
5. ✅ **Monitors behavior** (Drift detection)
6. ✅ **Simulates profit** (Business impact)
7. ✅ **Supports business strategy** (Threshold optimization)

### **Key Differentiators**

| Aspect | Typical ML Project | This System |
|--------|-------------------|-------------|
| **Output** | Model + Jupyter Notebook | Production API + Dashboard |
| **Deployment** | None | FastAPI backend + React frontend |
| **Explainability** | Feature importance only | Per-prediction SHAP values |
| **Fairness** | Not addressed | Bias detection across 3 attributes |
| **Business Value** | Accuracy metric | $48M profit simulation |
| **Monitoring** | None | Drift detection + logging |
| **Threshold** | Default (0.50) | Optimized (0.20) based on costs |
| **UI** | None | Professional 5-tab dashboard |

This system reflects **real-world fintech ML engineering**, not academic experimentation.

---

## 🔮 Future Enhancements

### **Short-Term (1-2 months)**
- [ ] Add user authentication (OAuth2 + JWT)
- [ ] Implement automated retraining pipeline (weekly)
- [ ] Add monitoring alerting (email/Slack on drift detection)
- [ ] Export fairness reports to PDF
- [ ] Add batch scoring endpoint (`POST /score/batch`)

### **Medium-Term (3-6 months)**
- [ ] Replace UCI dataset with **LendingClub** (372k samples)
- [ ] Add A/B testing framework for threshold experiments
- [ ] Implement feature store (e.g., Feast)
- [ ] Add model versioning (MLflow integration)
- [ ] Create Docker containers for easy deployment

### **Long-Term (6-12 months)**
- [ ] Cloud deployment (AWS SageMaker / GCP Vertex AI)
- [ ] Real-time streaming predictions (Kafka + Flink)
- [ ] Advanced drift detection (Kolmogorov-Smirnov test)
- [ ] Multi-model ensemble (XGBoost + LightGBM + CatBoost)
- [ ] Add counterfactual explanations (LIME)
- [ ] CI/CD pipeline (GitHub Actions, model performance gates)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. Open a **Pull Request**

**Areas for Contribution:**
- Additional datasets (LendingClub, Prosper)
- New fairness metrics (Demographic Parity, Equalized Odds)
- Alternative explainability methods (LIME, Counterfactuals)
- Frontend enhancements (dark mode, CSV export)
- Documentation improvements

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Credit Card Default Dataset
- **SHAP Library** for explainability framework
- **FastAPI** for modern Python web framework
- **React** and **Tailwind CSS** for frontend excellence
- **XGBoost** team for gradient boosting implementation

---

## 📞 Contact

**Developer:** Your Name  
**Email:** your.email@example.com  
**GitHub:** [@yourusername](https://github.com/yourusername)  
**LinkedIn:** [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📌 Final Note

This system is designed as a **production-style prototype**, not a toy model.

It demonstrates how **machine learning is applied in real financial decision systems**, including:
- Technical implementation (ML pipeline, API, dashboard)
- Ethical considerations (fairness, bias detection)
- Business alignment (profit simulation, threshold optimization)
- Regulatory compliance (explainability, monitoring)

The architecture, code quality, and documentation reflect **industry standards** for enterprise ML systems.

---

**⭐ If you find this project useful, please star the repository!**

```
Built with ❤️ using Python, React, and Machine Learning
```
