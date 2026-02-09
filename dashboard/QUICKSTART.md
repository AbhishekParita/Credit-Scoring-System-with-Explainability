# Dashboard Quick Start Guide

## 🚀 Setup (5 minutes)

### 1. Install Node.js
Download from: https://nodejs.org/ (use LTS version 16+)

### 2. Install Dashboard Dependencies

```bash
cd dashboard
npm install
```

*This will install React, Tailwind CSS, Recharts, and all dependencies*

### 3. Start Backend API

```bash
# In a separate terminal, from project root
cd ..
.\venv\Scripts\activate
uvicorn src.api:app --reload
```

✅ Backend running at: http://localhost:8000

### 4. Start Dashboard

```bash
# In dashboard folder
npm start
```

✅ Dashboard opens at: http://localhost:3000

---

## 📊 What You'll See

### Tab 1: Credit Scoring
- Form with all UCI dataset fields
- "Score Application" button → calls POST /score
- Results show: probability, decision, credit score, SHAP chart

### Tab 2: Portfolio
- Current threshold, approval rate, predictions
- Expected profit metrics

### Tab 3: Business Simulation
- Select threshold (0.2, 0.3, 0.5)
- See profit/loss breakdown
- Line chart: profit vs threshold

### Tab 4: Fairness
- Tabs: SEX, EDUCATION, MARRIAGE
- Bar charts for recall, approval rate, FNR
- Disparity analysis table

### Tab 5: Monitoring
- Drift detection for max_delay feature
- Approval rate trend
- Status: NORMAL/WARNING

---

## 🛠️ Troubleshooting

**Dashboard won't start:**
```bash
npm install
npm start
```

**API errors:**
- Check backend is running (port 8000)
- Run: `python run_fairness_analysis.py` (for fairness tab)

**Styles look wrong:**
```bash
# Restart dev server
Ctrl+C
npm start
```

---

## 🎨 Theme Colors

- **Sidebar:** Dark slate (#0F172A)
- **SHAP Green:** Emerald (#10B981) = reduces risk
- **SHAP Red:** Rose (#F43F5E) = increases risk
- **Primary:** Blue gradients with glow effects

---

## 📦 What's Included

✅ 5 complete dashboard tabs  
✅ Real API integration (no fake data)  
✅ Professional fintech design  
✅ Responsive charts  
✅ Modern Tailwind styling  

**Total files:** 10 files, ~1,400 lines of React code
