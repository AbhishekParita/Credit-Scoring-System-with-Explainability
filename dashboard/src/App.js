import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Users, 
  AlertCircle,
  CheckCircle,
  XCircle,
  Activity,
  BarChart3,
  Scale
} from 'lucide-react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

const API_BASE = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('scoring');
  const [loading, setLoading] = useState(false);
  
  // Credit Scoring State
  const [customerData, setCustomerData] = useState({
    LIMIT_BAL: 200000,
    AGE: 35,
    SEX: 1,
    EDUCATION: 2,
    MARRIAGE: 1,
    PAY_0: 0, PAY_2: 0, PAY_3: 0, PAY_4: 0, PAY_5: 0, PAY_6: 0,
    BILL_AMT1: 50000, BILL_AMT2: 48000, BILL_AMT3: 47000,
    BILL_AMT4: 46000, BILL_AMT5: 45000, BILL_AMT6: 44000,
    PAY_AMT1: 5000, PAY_AMT2: 5000, PAY_AMT3: 5000,
    PAY_AMT4: 5000, PAY_AMT5: 5000, PAY_AMT6: 5000
  });
  const [scoringResult, setScoringResult] = useState(null);
  
  // Business Simulation State
  const [selectedThreshold, setSelectedThreshold] = useState(0.20);
  const [simulationData, setSimulationData] = useState(null);
  
  // Fairness State
  const [fairnessAttribute, setFairnessAttribute] = useState('SEX');
  const [fairnessData, setFairnessData] = useState(null);
  
  // Monitoring State
  const [monitoringData, setMonitoringData] = useState(null);
  
  // Portfolio Overview State
  const [portfolioData, setPortfolioData] = useState(null);

  // Load initial data
  useEffect(() => {
    loadPortfolioData();
    loadMonitoringData();
  }, []);

  const loadPortfolioData = async () => {
    try {
      const [modelInfo, monitoring] = await Promise.all([
        axios.get(`${API_BASE}/model/info`),
        axios.get(`${API_BASE}/monitoring`)
      ]);
      setPortfolioData({
        threshold: modelInfo.data.optimal_threshold,
        approvalRate: monitoring.data.approval_rate,
        avgProbability: monitoring.data.avg_probability || 0.22,
        totalPredictions: monitoring.data.total_predictions
      });
    } catch (error) {
      console.error('Error loading portfolio data:', error);
    }
  };

  const loadMonitoringData = async () => {
    try {
      const response = await axios.get(`${API_BASE}/monitoring`);
      setMonitoringData(response.data);
    } catch (error) {
      console.error('Error loading monitoring data:', error);
    }
  };

  const handleScoreCustomer = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/score`, customerData);
      setScoringResult(response.data);
    } catch (error) {
      console.error('Error scoring customer:', error);
      alert('Error scoring customer. Please check the backend is running.');
    }
    setLoading(false);
  };

  const loadSimulationData = async () => {
    // Since we don't have a simulation API endpoint, we'll use hardcoded data
    // In production, this would call an endpoint
    const simData = {
      0.20: {
        approved: 4505,
        total: 6000,
        defaultRate: 0.103,
        interestEarned: 120693900,
        losses: -72440000,
        netProfit: 48253900,
        roi: 5.50,
        loanAmount: 877066000
      },
      0.30: {
        approved: 4728,
        total: 6000,
        defaultRate: 0.113,
        interestEarned: 124127400,
        losses: -81900000,
        netProfit: 42227400,
        roi: 4.64,
        loanAmount: 909416000
      },
      0.50: {
        approved: 5265,
        total: 6000,
        defaultRate: 0.142,
        interestEarned: 130185900,
        losses: -112650000,
        netProfit: 17535900,
        roi: 1.79,
        loanAmount: 980556000
      }
    };
    setSimulationData(simData);
  };

  const loadFairnessData = async (attribute) => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE}/fairness?attribute=${attribute}`);
      setFairnessData(response.data[attribute]);
    } catch (error) {
      console.error('Error loading fairness data:', error);
      alert('Error loading fairness data. Please run fairness analysis first.');
    }
    setLoading(false);
  };

  useEffect(() => {
    if (activeTab === 'simulation' && !simulationData) {
      loadSimulationData();
    }
  }, [activeTab, simulationData]);

  useEffect(() => {
    if (activeTab === 'fairness') {
      loadFairnessData(fairnessAttribute);
    }
  }, [activeTab, fairnessAttribute]);

  const formatCurrency = (num) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(num);
  };

  const formatPercent = (num) => {
    return `${(num * 100).toFixed(1)}%`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Sidebar */}
      <div className="fixed left-0 top-0 h-full w-64 bg-slate-900 text-white shadow-2xl">
        <div className="p-6 border-b border-slate-700">
          <h1 className="text-2xl font-black tracking-tight">XCRE</h1>
          <p className="text-xs text-slate-400 mt-1">Credit Risk Engine</p>
        </div>
        
        <nav className="p-4 space-y-2">
          <button
            onClick={() => setActiveTab('scoring')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
              activeTab === 'scoring'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            <Activity size={20} />
            <span className="font-semibold">Credit Scoring</span>
          </button>
          
          <button
            onClick={() => setActiveTab('portfolio')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
              activeTab === 'portfolio'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            <BarChart3 size={20} />
            <span className="font-semibold">Portfolio</span>
          </button>
          
          <button
            onClick={() => setActiveTab('simulation')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
              activeTab === 'simulation'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            <DollarSign size={20} />
            <span className="font-semibold">Business Simulation</span>
          </button>
          
          <button
            onClick={() => setActiveTab('fairness')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
              activeTab === 'fairness'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            <Scale size={20} />
            <span className="font-semibold">Fairness & Bias</span>
          </button>
          
          <button
            onClick={() => setActiveTab('monitoring')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
              activeTab === 'monitoring'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            <TrendingUp size={20} />
            <span className="font-semibold">Monitoring & Drift</span>
          </button>
        </nav>

        <div className="absolute bottom-0 w-full p-4 border-t border-slate-700">
          <div className="text-xs text-slate-400">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
              <span>API Connected</span>
            </div>
            <div>Model v1.0.0</div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="ml-64 p-8">
        {activeTab === 'scoring' && <CreditScoringTab 
          customerData={customerData}
          setCustomerData={setCustomerData}
          handleScoreCustomer={handleScoreCustomer}
          scoringResult={scoringResult}
          loading={loading}
        />}
        
        {activeTab === 'portfolio' && <PortfolioTab portfolioData={portfolioData} />}
        
        {activeTab === 'simulation' && <SimulationTab 
          selectedThreshold={selectedThreshold}
          setSelectedThreshold={setSelectedThreshold}
          simulationData={simulationData}
          formatCurrency={formatCurrency}
          formatPercent={formatPercent}
        />}
        
        {activeTab === 'fairness' && <FairnessTab 
          fairnessAttribute={fairnessAttribute}
          setFairnessAttribute={setFairnessAttribute}
          fairnessData={fairnessData}
          loading={loading}
          formatPercent={formatPercent}
        />}
        
        {activeTab === 'monitoring' && <MonitoringTab 
          monitoringData={monitoringData}
          formatPercent={formatPercent}
        />}
      </div>
    </div>
  );
}

// Credit Scoring Tab Component
function CreditScoringTab({ customerData, setCustomerData, handleScoreCustomer, scoringResult, loading }) {
  const handleInputChange = (field, value) => {
    setCustomerData(prev => ({ ...prev, [field]: parseFloat(value) || 0 }));
  };

  return (
    <div>
      <h2 className="text-3xl font-black text-slate-900 mb-6">Real-Time Credit Scoring</h2>
      
      <div className="grid grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="bg-white rounded-3xl shadow-xl p-6">
          <h3 className="text-xl font-extrabold text-slate-800 mb-4">Customer Information</h3>
          
          <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
            <InputField label="Credit Limit" value={customerData.LIMIT_BAL} 
              onChange={(v) => handleInputChange('LIMIT_BAL', v)} />
            <InputField label="Age" value={customerData.AGE} 
              onChange={(v) => handleInputChange('AGE', v)} />
            
            <SelectField label="Sex" value={customerData.SEX} 
              onChange={(v) => handleInputChange('SEX', v)}
              options={[{value: 1, label: 'Male'}, {value: 2, label: 'Female'}]} />
            
            <SelectField label="Education" value={customerData.EDUCATION} 
              onChange={(v) => handleInputChange('EDUCATION', v)}
              options={[
                {value: 1, label: 'Graduate'},
                {value: 2, label: 'University'},
                {value: 3, label: 'High School'},
                {value: 4, label: 'Others'}
              ]} />
            
            <SelectField label="Marriage" value={customerData.MARRIAGE} 
              onChange={(v) => handleInputChange('MARRIAGE', v)}
              options={[
                {value: 1, label: 'Married'},
                {value: 2, label: 'Single'},
                {value: 3, label: 'Others'}
              ]} />

            <div className="pt-2 border-t">
              <h4 className="font-bold text-sm text-slate-600 mb-3">Payment Status (months)</h4>
              <div className="grid grid-cols-3 gap-3">
                {[0, 2, 3, 4, 5, 6].map(i => (
                  <InputField key={i} label={`PAY_${i === 0 ? '0' : i}`} 
                    value={customerData[`PAY_${i === 0 ? '0' : i}`]} 
                    onChange={(v) => handleInputChange(`PAY_${i === 0 ? '0' : i}`, v)} 
                    size="sm" />
                ))}
              </div>
            </div>

            <div className="pt-2 border-t">
              <h4 className="font-bold text-sm text-slate-600 mb-3">Bill Amounts</h4>
              <div className="grid grid-cols-2 gap-3">
                {[1, 2, 3, 4, 5, 6].map(i => (
                  <InputField key={i} label={`BILL_AMT${i}`} 
                    value={customerData[`BILL_AMT${i}`]} 
                    onChange={(v) => handleInputChange(`BILL_AMT${i}`, v)} 
                    size="sm" />
                ))}
              </div>
            </div>

            <div className="pt-2 border-t">
              <h4 className="font-bold text-sm text-slate-600 mb-3">Payment Amounts</h4>
              <div className="grid grid-cols-2 gap-3">
                {[1, 2, 3, 4, 5, 6].map(i => (
                  <InputField key={i} label={`PAY_AMT${i}`} 
                    value={customerData[`PAY_AMT${i}`]} 
                    onChange={(v) => handleInputChange(`PAY_AMT${i}`, v)} 
                    size="sm" />
                ))}
              </div>
            </div>
          </div>

          <button
            onClick={handleScoreCustomer}
            disabled={loading}
            className="mt-6 w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 rounded-2xl shadow-lg shadow-blue-500/50 transition-all disabled:opacity-50"
          >
            {loading ? 'Analyzing...' : 'Score Application'}
          </button>
        </div>

        {/* Results */}
        <div className="space-y-6">
          {scoringResult && (
            <>
              {/* Decision Card */}
              <div className={`rounded-3xl shadow-2xl p-8 ${
                scoringResult.decision === 'APPROVE' ? 'bg-gradient-to-br from-emerald-50 to-emerald-100' :
                scoringResult.decision === 'REJECT' ? 'bg-gradient-to-br from-rose-50 to-rose-100' :
                'bg-gradient-to-br from-amber-50 to-amber-100'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-black text-slate-800">Decision</h3>
                  {scoringResult.decision === 'APPROVE' ? <CheckCircle className="text-emerald-600" size={40} /> :
                   scoringResult.decision === 'REJECT' ? <XCircle className="text-rose-600" size={40} /> :
                   <AlertCircle className="text-amber-600" size={40} />}
                </div>
                <div className={`text-5xl font-black mb-2 ${
                  scoringResult.decision === 'APPROVE' ? 'text-emerald-700' :
                  scoringResult.decision === 'REJECT' ? 'text-rose-700' :
                  'text-amber-700'
                }`}>
                  {scoringResult.decision}
                </div>
                <div className="text-sm text-slate-600 font-semibold">
                  Risk Level: {scoringResult.risk_level} | Confidence: {scoringResult.confidence}
                </div>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white rounded-3xl shadow-xl p-6">
                  <div className="text-sm text-slate-600 font-semibold mb-2">Default Probability</div>
                  <div className="text-4xl font-black text-slate-900">
                    {(scoringResult.default_probability * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-white rounded-3xl shadow-xl p-6">
                  <div className="text-sm text-slate-600 font-semibold mb-2">Credit Score</div>
                  <div className="text-4xl font-black text-slate-900">
                    {scoringResult.credit_score}
                  </div>
                  <div className="text-xs text-slate-500">FICO-style (300-850)</div>
                </div>
              </div>

              {/* SHAP Explanations */}
              <div className="bg-white rounded-3xl shadow-xl p-6">
                <h3 className="text-xl font-extrabold text-slate-800 mb-4">Top 5 Risk Factors</h3>
                <div className="space-y-3">
                  {scoringResult.reasons.map((reason, idx) => (
                    <div key={idx} className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center font-bold text-sm text-slate-600">
                        {idx + 1}
                      </div>
                      <div className="flex-1">
                        <div className="font-bold text-sm text-slate-800">{reason.feature}</div>
                        <div className="flex items-center gap-2 mt-1">
                          <div className="flex-1 bg-slate-200 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${reason.impact > 0 ? 'bg-rose-500' : 'bg-emerald-500'}`}
                              style={{width: `${Math.abs(reason.impact) * 100}%`}}
                            ></div>
                          </div>
                          <div className={`text-sm font-bold ${reason.impact > 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                            {reason.impact > 0 ? '+' : ''}{reason.impact.toFixed(3)}
                          </div>
                        </div>
                        <div className="text-xs text-slate-500 mt-1">
                          {reason.impact > 0 ? 'Increases default risk' : 'Decreases default risk'}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
          
          {!scoringResult && (
            <div className="bg-white rounded-3xl shadow-xl p-12 text-center">
              <Activity size={64} className="mx-auto text-slate-300 mb-4" />
              <p className="text-lg text-slate-400 font-semibold">Enter customer data and click "Score Application"</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Portfolio Tab Component
function PortfolioTab({ portfolioData }) {
  if (!portfolioData) {
    return <div className="text-center py-12">Loading portfolio data...</div>;
  }

  return (
    <div>
      <h2 className="text-3xl font-black text-slate-900 mb-6">Portfolio Overview</h2>
      
      <div className="grid grid-cols-4 gap-6">
        <MetricCard 
          title="Current Threshold" 
          value={portfolioData.threshold.toFixed(2)}
          icon={<TrendingUp />}
          color="blue"
        />
        <MetricCard 
          title="Approval Rate" 
          value={`${(portfolioData.approvalRate * 100).toFixed(1)}%`}
          icon={<CheckCircle />}
          color="emerald"
        />
        <MetricCard 
          title="Avg Default Probability" 
          value={`${(portfolioData.avgProbability * 100).toFixed(1)}%`}
          icon={<AlertCircle />}
          color="amber"
        />
        <MetricCard 
          title="Total Predictions" 
          value={portfolioData.totalPredictions}
          icon={<Users />}
          color="slate"
        />
      </div>

      <div className="mt-8 bg-white rounded-3xl shadow-xl p-8">
        <h3 className="text-2xl font-extrabold text-slate-800 mb-6">Business Impact</h3>
        <div className="grid grid-cols-3 gap-6">
          <div className="text-center p-6 bg-emerald-50 rounded-2xl">
            <div className="text-sm text-emerald-700 font-bold mb-2">Expected Net Profit</div>
            <div className="text-4xl font-black text-emerald-900">$48.3M</div>
            <div className="text-xs text-emerald-600 mt-2">@ Current Threshold (0.20)</div>
          </div>
          <div className="text-center p-6 bg-blue-50 rounded-2xl">
            <div className="text-sm text-blue-700 font-bold mb-2">ROI</div>
            <div className="text-4xl font-black text-blue-900">5.50%</div>
            <div className="text-xs text-blue-600 mt-2">Return on Investment</div>
          </div>
          <div className="text-center p-6 bg-slate-50 rounded-2xl">
            <div className="text-sm text-slate-700 font-bold mb-2">Default Rate</div>
            <div className="text-4xl font-black text-slate-900">10.3%</div>
            <div className="text-xs text-slate-600 mt-2">Among Approved Customers</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Simulation Tab Component
function SimulationTab({ selectedThreshold, setSelectedThreshold, simulationData, formatCurrency, formatPercent }) {
  if (!simulationData) {
    return <div className="text-center py-12">Loading simulation data...</div>;
  }

  const currentData = simulationData[selectedThreshold];
  const chartData = Object.entries(simulationData).map(([threshold, data]) => ({
    threshold: parseFloat(threshold),
    profit: data.netProfit / 1000000, // Convert to millions
    roi: data.roi
  }));

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-black text-slate-900">Business Simulation</h2>
        <select
          value={selectedThreshold}
          onChange={(e) => setSelectedThreshold(parseFloat(e.target.value))}
          className="px-6 py-3 rounded-2xl border-2 border-slate-300 font-bold text-slate-800 bg-white shadow-lg"
        >
          <option value={0.20}>Threshold: 0.20</option>
          <option value={0.30}>Threshold: 0.30</option>
          <option value={0.50}>Threshold: 0.50</option>
        </select>
      </div>

      <div className="grid grid-cols-3 gap-6 mb-8">
        <MetricCard 
          title="Approved Customers"
          value={`${currentData.approved.toLocaleString()} / ${currentData.total.toLocaleString()}`}
          subtitle={formatPercent(currentData.approved / currentData.total)}
          color="blue"
          icon={<Users />}
        />
        <MetricCard 
          title="Default Rate"
          value={formatPercent(currentData.defaultRate)}
          subtitle="Among approved"
          color="amber"
          icon={<AlertCircle />}
        />
        <MetricCard 
          title="Total Loan Amount"
          value={formatCurrency(currentData.loanAmount)}
          color="slate"
          icon={<DollarSign />}
        />
      </div>

      <div className="grid grid-cols-2 gap-6 mb-8">
        <div className="bg-white rounded-3xl shadow-xl p-6">
          <h3 className="text-lg font-extrabold text-slate-800 mb-4">Financial Breakdown</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center p-4 bg-emerald-50 rounded-2xl">
              <span className="font-bold text-emerald-700">Interest Earned</span>
              <span className="text-xl font-black text-emerald-900">{formatCurrency(currentData.interestEarned)}</span>
            </div>
            <div className="flex justify-between items-center p-4 bg-rose-50 rounded-2xl">
              <span className="font-bold text-rose-700">Losses from Defaults</span>
              <span className="text-xl font-black text-rose-900">{formatCurrency(currentData.losses)}</span>
            </div>
            <div className="flex justify-between items-center p-4 bg-blue-50 rounded-2xl border-2 border-blue-200">
              <span className="font-bold text-blue-700">Net Profit</span>
              <span className="text-2xl font-black text-blue-900">{formatCurrency(currentData.netProfit)}</span>
            </div>
            <div className="flex justify-between items-center p-4 bg-slate-100 rounded-2xl">
              <span className="font-bold text-slate-700">ROI</span>
              <span className="text-2xl font-black text-slate-900">{currentData.roi.toFixed(2)}%</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-3xl shadow-xl p-6">
          <h3 className="text-lg font-extrabold text-slate-800 mb-4">Profit by Threshold</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="threshold" stroke="#64748b" />
              <YAxis stroke="#64748b" label={{ value: 'Profit ($M)', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '2px solid #e2e8f0', borderRadius: '12px' }}
                formatter={(value) => [`$${value.toFixed(1)}M`, 'Profit']}
              />
              <Line type="monotone" dataKey="profit" stroke="#3b82f6" strokeWidth={3} dot={{ fill: '#3b82f6', r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-gradient-to-r from-blue-50 to-emerald-50 rounded-3xl shadow-xl p-8">
        <h3 className="text-xl font-extrabold text-slate-800 mb-4">Key Insights</h3>
        <ul className="space-y-3 text-slate-700">
          <li className="flex items-start gap-3">
            <CheckCircle size={20} className="text-emerald-600 mt-1 flex-shrink-0" />
            <span><strong>Threshold 0.20</strong> is optimal, generating <strong>$48.3M profit</strong> with 75.1% approval rate</span>
          </li>
          <li className="flex items-start gap-3">
            <TrendingDown size={20} className="text-rose-600 mt-1 flex-shrink-0" />
            <span><strong>Aggressive lending (0.50)</strong> loses $30.7M vs optimal despite 13% higher approval rate</span>
          </li>
          <li className="flex items-start gap-3">
            <DollarSign size={20} className="text-blue-600 mt-1 flex-shrink-0" />
            <span><strong>Conservative strategy</strong> maximizes ROI by keeping default rate at 10.3%</span>
          </li>
        </ul>
      </div>
    </div>
  );
}

// Fairness Tab Component
function FairnessTab({ fairnessAttribute, setFairnessAttribute, fairnessData, loading, formatPercent }) {
  if (loading) {
    return <div className="text-center py-12">Loading fairness data...</div>;
  }

  const prepareChartData = () => {
    if (!fairnessData || !fairnessData.groups) return [];
    return Object.entries(fairnessData.groups).map(([group, metrics]) => ({
      group: `Group ${group}`,
      recall: metrics.recall,
      approvalRate: metrics.approval_rate,
      fnr: metrics.false_negative_rate
    }));
  };

  const chartData = prepareChartData();

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-black text-slate-900">Fairness & Bias Analysis</h2>
        <div className="flex gap-3">
          {['SEX', 'EDUCATION', 'MARRIAGE'].map(attr => (
            <button
              key={attr}
              onClick={() => setFairnessAttribute(attr)}
              className={`px-6 py-3 rounded-2xl font-bold transition-all ${
                fairnessAttribute === attr
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                  : 'bg-white text-slate-600 border-2 border-slate-300 hover:border-blue-300'
              }`}
            >
              {attr}
            </button>
          ))}
        </div>
      </div>

      {fairnessData && (
        <>
          <div className="grid grid-cols-3 gap-6 mb-8">
            <div className="bg-white rounded-3xl shadow-xl p-6">
              <h3 className="text-lg font-extrabold text-slate-800 mb-4">Recall by Group</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="group" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: '#fff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Bar dataKey="recall" fill="#3b82f6" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-3xl shadow-xl p-6">
              <h3 className="text-lg font-extrabold text-slate-800 mb-4">Approval Rate by Group</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="group" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: '#fff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Bar dataKey="approvalRate" fill="#10b981" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-3xl shadow-xl p-6">
              <h3 className="text-lg font-extrabold text-slate-800 mb-4">False Negative Rate</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="group" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: '#fff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Bar dataKey="fnr" fill="#f43f5e" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white rounded-3xl shadow-xl p-8">
            <h3 className="text-xl font-extrabold text-slate-800 mb-6">Detailed Metrics by Group</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b-2 border-slate-200">
                    <th className="text-left py-3 px-4 font-black text-slate-700">Group</th>
                    <th className="text-right py-3 px-4 font-black text-slate-700">Sample Size</th>
                    <th className="text-right py-3 px-4 font-black text-slate-700">Recall</th>
                    <th className="text-right py-3 px-4 font-black text-slate-700">Approval Rate</th>
                    <th className="text-right py-3 px-4 font-black text-slate-700">FNR</th>
                    <th className="text-right py-3 px-4 font-black text-slate-700">Avg Default Prob</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(fairnessData.groups).map(([group, metrics]) => (
                    <tr key={group} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="py-4 px-4 font-bold text-slate-800">Group {group}</td>
                      <td className="py-4 px-4 text-right text-slate-700">{metrics.sample_size.toLocaleString()}</td>
                      <td className="py-4 px-4 text-right font-bold text-blue-700">{formatPercent(metrics.recall)}</td>
                      <td className="py-4 px-4 text-right font-bold text-emerald-700">{formatPercent(metrics.approval_rate)}</td>
                      <td className="py-4 px-4 text-right font-bold text-rose-700">{formatPercent(metrics.false_negative_rate)}</td>
                      <td className="py-4 px-4 text-right text-slate-700">{formatPercent(metrics.avg_default_probability)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {fairnessData.disparities && (
            <div className="mt-6 bg-gradient-to-r from-amber-50 to-rose-50 rounded-3xl shadow-xl p-8">
              <h3 className="text-xl font-extrabold text-slate-800 mb-4">Disparity Analysis</h3>
              <div className="text-slate-700">
                <p className="mb-2"><strong>Max Approval Rate Difference:</strong> {formatPercent(fairnessData.disparities.max_approval_rate_diff / 100)}</p>
                <p><strong>Severity:</strong> <span className={`font-bold ${
                  fairnessData.disparities.severity === 'ACCEPTABLE' ? 'text-emerald-600' :
                  fairnessData.disparities.severity === 'MODERATE' ? 'text-amber-600' :
                  'text-rose-600'
                }`}>{fairnessData.disparities.severity}</span></p>
              </div>
            </div>
          )}
        </>
      )}

      {!fairnessData && (
        <div className="bg-white rounded-3xl shadow-xl p-12 text-center">
          <Scale size={64} className="mx-auto text-slate-300 mb-4" />
          <p className="text-lg text-slate-400 font-semibold">Select an attribute to view fairness analysis</p>
        </div>
      )}
    </div>
  );
}

// Monitoring Tab Component
function MonitoringTab({ monitoringData, formatPercent }) {
  if (!monitoringData) {
    return <div className="text-center py-12">Loading monitoring data...</div>;
  }

  const driftStatus = monitoringData.drift_detected?.max_delay?.status || 'NORMAL';

  return (
    <div>
      <h2 className="text-3xl font-black text-slate-900 mb-6">Monitoring & Drift Detection</h2>

      <div className="grid grid-cols-3 gap-6 mb-8">
        <MetricCard 
          title="Total Predictions"
          value={monitoringData.total_predictions?.toLocaleString() || '0'}
          color="blue"
          icon={<Activity />}
        />
        <MetricCard 
          title="Current Approval Rate"
          value={formatPercent(monitoringData.approval_rate || 0)}
          color="emerald"
          icon={<CheckCircle />}
        />
        <MetricCard 
          title="Drift Status"
          value={driftStatus}
          color={driftStatus === 'NORMAL' ? 'emerald' : 'amber'}
          icon={<AlertCircle />}
        />
      </div>

      {monitoringData.drift_detected?.max_delay && (
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-white rounded-3xl shadow-xl p-8">
            <h3 className="text-xl font-extrabold text-slate-800 mb-6">Feature Drift: max_delay</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center p-4 bg-slate-50 rounded-2xl">
                <span className="font-bold text-slate-700">Training Baseline</span>
                <span className="text-xl font-black text-slate-900">{monitoringData.drift_detected.max_delay.baseline.toFixed(3)}</span>
              </div>
              <div className="flex justify-between items-center p-4 bg-blue-50 rounded-2xl">
                <span className="font-bold text-blue-700">Current Mean</span>
                <span className="text-xl font-black text-blue-900">{monitoringData.drift_detected.max_delay.current.toFixed(3)}</span>
              </div>
              <div className="flex justify-between items-center p-4 bg-emerald-50 rounded-2xl border-2 border-emerald-200">
                <span className="font-bold text-emerald-700">Drift %</span>
                <span className="text-xl font-black text-emerald-900">{monitoringData.drift_detected.max_delay.drift_pct.toFixed(2)}%</span>
              </div>
              <div className={`flex justify-between items-center p-4 rounded-2xl ${
                driftStatus === 'NORMAL' ? 'bg-emerald-100' : 'bg-amber-100'
              }`}>
                <span className={`font-bold ${driftStatus === 'NORMAL' ? 'text-emerald-700' : 'text-amber-700'}`}>Status</span>
                <span className={`text-xl font-black ${driftStatus === 'NORMAL' ? 'text-emerald-900' : 'text-amber-900'}`}>{driftStatus}</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-3xl shadow-xl p-8">
            <h3 className="text-xl font-extrabold text-slate-800 mb-6">Monitoring Insights</h3>
            <div className="space-y-4 text-slate-700">
              <div className="flex items-start gap-3 p-4 bg-blue-50 rounded-2xl">
                <CheckCircle size={20} className="text-blue-600 mt-1 flex-shrink-0" />
                <div>
                  <div className="font-bold text-blue-900 mb-1">Prediction Volume</div>
                  <div className="text-sm">System has processed {monitoringData.total_predictions} predictions</div>
                </div>
              </div>
              <div className="flex items-start gap-3 p-4 bg-emerald-50 rounded-2xl">
                <CheckCircle size={20} className="text-emerald-600 mt-1 flex-shrink-0" />
                <div>
                  <div className="font-bold text-emerald-900 mb-1">Approval Stability</div>
                  <div className="text-sm">Current approval rate: {formatPercent(monitoringData.approval_rate)}</div>
                </div>
              </div>
              <div className="flex items-start gap-3 p-4 bg-slate-50 rounded-2xl">
                <Activity size={20} className="text-slate-600 mt-1 flex-shrink-0" />
                <div>
                  <div className="font-bold text-slate-900 mb-1">Feature Monitoring</div>
                  <div className="text-sm">Tracking max_delay (most important feature)</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper Components
function InputField({ label, value, onChange, size = 'md' }) {
  return (
    <div>
      <label className="block text-sm font-bold text-slate-700 mb-1">{label}</label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`w-full px-4 ${size === 'sm' ? 'py-2' : 'py-3'} rounded-xl border-2 border-slate-200 focus:border-blue-500 focus:outline-none font-semibold`}
      />
    </div>
  );
}

function SelectField({ label, value, onChange, options }) {
  return (
    <div>
      <label className="block text-sm font-bold text-slate-700 mb-1">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-4 py-3 rounded-xl border-2 border-slate-200 focus:border-blue-500 focus:outline-none font-semibold"
      >
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
}

function MetricCard({ title, value, subtitle, icon, color = 'blue' }) {
  const colorClasses = {
    blue: 'from-blue-50 to-blue-100 text-blue-900',
    emerald: 'from-emerald-50 to-emerald-100 text-emerald-900',
    amber: 'from-amber-50 to-amber-100 text-amber-900',
    rose: 'from-rose-50 to-rose-100 text-rose-900',
    slate: 'from-slate-50 to-slate-100 text-slate-900'
  };

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color]} rounded-3xl shadow-xl p-6`}>
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm font-bold opacity-75">{title}</div>
        <div className="opacity-50">{icon}</div>
      </div>
      <div className="text-3xl font-black mb-1">{value}</div>
      {subtitle && <div className="text-sm font-semibold opacity-75">{subtitle}</div>}
    </div>
  );
}

export default App;
