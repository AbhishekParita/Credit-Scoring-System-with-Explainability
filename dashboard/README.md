# Credit Scoring System Dashboard

Modern fintech dashboard for the Explainable Credit Risk Engine (XCRE). Built with React, Tailwind CSS, and Recharts.

## Features

### 1. Real-Time Credit Scoring
- Interactive form for customer data input (19 UCI dataset fields)
- Live API integration with FastAPI backend
- Display default probability, credit score, and decision
- Top 5 SHAP explanations with visual impact indicators
- Color-coded decisions (Approve/Reject/Manual Review)

### 2. Portfolio Overview
- Current model threshold
- Overall approval rate
- Average default probability
- Total predictions processed
- Expected profit metrics from business simulation

### 3. Business Simulation
- Threshold selector (0.2, 0.3, 0.5)
- Financial breakdown (interest earned, losses, net profit, ROI)
- Interactive profit vs threshold line chart
- Comparison of lending strategies

### 4. Fairness & Bias Analysis
- Tabs for SEX, EDUCATION, MARRIAGE attributes
- Per-group metrics (recall, approval rate, FNR)
- Bar charts for visual group comparison
- Disparity analysis with severity flags

### 5. Monitoring & Drift Detection
- Total prediction volume
- Current approval rate
- Feature drift detection (max_delay)
- Drift status indicators (NORMAL/WARNING)

## Tech Stack

- **React 18** - UI framework
- **Tailwind CSS** - Modern fintech styling
- **Recharts** - Data visualization
- **Axios** - API communication
- **Lucide React** - Icon library

## Design Theme: Modern Fintech

- **Sidebar:** Slate 900 (#0F172A) for institutional trust
- **Typography:** Heavy weights (font-black, font-extrabold) for metrics
- **Semantic Feedback:**
  - Emerald (#10B981) for risk-lowering factors
  - Rose (#F43F5E) for risk-increasing factors
  - Blue gradients for primary actions
- **Cards:** 3xl border-radius with soft shadows for premium feel

## Prerequisites

1. **Node.js** - Version 16 or higher
2. **npm** or **yarn** package manager
3. **FastAPI Backend** - Running on http://localhost:8000

## Installation

### Step 1: Install Dependencies

```bash
cd dashboard
npm install
```

### Step 2: Verify Backend is Running

Make sure your FastAPI backend is running:

```bash
# In the root project directory
cd ..
.\venv\Scripts\activate
uvicorn src.api:app --reload
```

The backend should be accessible at: http://localhost:8000

### Step 3: Start the Dashboard

```bash
npm start
```

The dashboard will open automatically at: http://localhost:3000

## API Endpoints Used

The dashboard connects to these FastAPI endpoints:

- `GET /` - Health check
- `POST /score` - Score customer application
- `GET /model/info` - Model metrics
- `GET /monitoring` - Monitoring stats and drift
- `GET /fairness?attribute=<name>` - Fairness analysis

## Project Structure

```
dashboard/
├── public/
│   └── index.html              # HTML template
├── src/
│   ├── App.js                  # Main application component
│   ├── index.js                # React entry point
│   └── index.css               # Tailwind CSS imports
├── package.json                # Dependencies
├── tailwind.config.js          # Tailwind configuration
└── postcss.config.js           # PostCSS configuration
```

## Usage

### 1. Score a Customer

1. Navigate to "Credit Scoring" tab
2. Fill in customer information:
   - Credit Limit (LIMIT_BAL)
   - Age, Sex, Education, Marriage
   - Payment history (PAY_0 through PAY_6)
   - Bill amounts (BILL_AMT1-6)
   - Payment amounts (PAY_AMT1-6)
3. Click "Score Application"
4. View decision, probability, credit score, and SHAP explanations

### 2. View Portfolio Metrics

1. Navigate to "Portfolio" tab
2. See current model configuration
3. View approval rate and default probability
4. Check expected business metrics

### 3. Simulate Business Outcomes

1. Navigate to "Business Simulation" tab
2. Select threshold from dropdown (0.20, 0.30, 0.50)
3. View:
   - Approved customer count
   - Default rate among approved
   - Interest earned vs losses
   - Net profit and ROI
4. Compare profit trends across thresholds

### 4. Analyze Fairness

1. Navigate to "Fairness & Bias" tab
2. Select attribute (SEX, EDUCATION, MARRIAGE)
3. View bar charts for:
   - Recall by group
   - Approval rate by group
   - False negative rate by group
4. Review detailed metrics table
5. Check disparity analysis

### 5. Monitor Drift

1. Navigate to "Monitoring & Drift" tab
2. View total predictions processed
3. Check current approval rate
4. Review max_delay feature drift
5. Monitor drift status (NORMAL/WARNING)

## Customization

### Changing API Base URL

If your backend runs on a different port:

```javascript
// In src/App.js, line 28
const API_BASE = 'http://localhost:YOUR_PORT';
```

### Modifying Thresholds

To test different thresholds in Business Simulation:

```javascript
// In src/App.js, loadSimulationData() function
// Add new threshold data matching your backend results
```

### Adjusting Colors

Edit `tailwind.config.js` to customize the color palette:

```javascript
extend: {
  colors: {
    primary: {...},
    secondary: {...}
  }
}
```

## Troubleshooting

### Issue: "API Connected" shows red dot

**Solution:** Backend is not running. Start FastAPI:
```bash
uvicorn src.api:app --reload
```

### Issue: Fairness data shows error

**Solution:** Run fairness analysis first:
```bash
python run_fairness_analysis.py
```

### Issue: Blank screen after npm start

**Solution:** Check browser console (F12) for errors. Verify all dependencies installed:
```bash
npm install
```

### Issue: Tailwind styles not loading

**Solution:** Restart development server:
```bash
# Stop server (Ctrl+C)
npm start
```

## Production Build

To create an optimized production build:

```bash
npm run build
```

This creates a `build/` folder with static files ready for deployment.

### Serving Production Build

```bash
npx serve -s build -l 3000
```

## Performance Notes

- Dashboard loads data on tab change (lazy loading)
- API calls are debounced to prevent excessive requests
- Charts use ResponsiveContainer for optimal rendering
- Icon library is tree-shaken (only used icons imported)

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Future Enhancements

- Real-time prediction streaming (WebSocket)
- Historical trend charts for monitoring
- Export reports to PDF
- Custom threshold simulation ranges
- User authentication

## License

Part of the Explainable Credit Risk Engine (XCRE) project.

---

**Built with React ⚛️ | Styled with Tailwind 🎨 | Powered by FastAPI 🚀**
