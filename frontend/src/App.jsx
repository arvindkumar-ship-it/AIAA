import { useEffect, useState } from 'react'
import {
  Brain, Bell, UserCog, ShieldCheck, Route, Activity,
  Send, Loader2, CheckCircle2, AlertTriangle, XCircle,
} from 'lucide-react'
import './App.css'
import { createTask, getMetrics, BASE_URL } from './api'

const MODULES = [
  {
    key: 'memory',
    name: 'Memory Optimization',
    icon: Brain,
    desc: 'AgeMem + SimpleMem unified STM/LTM policy — entropy-based compression on long-horizon context.',
    metric: 'Compression',
    value: '70%',
  },
  {
    key: 'intervention',
    name: 'Intervention Predictor',
    icon: Bell,
    desc: 'Calibrated LSTM-style classifier deciding exactly when a human ping is worth interrupting for.',
    metric: 'PTS',
    value: '0.78',
  },
  {
    key: 'style',
    name: 'Style Classifier',
    icon: UserCog,
    desc: 'PATHs-based Random Forest reading collaboration style from a user\'s intervention history.',
    metric: 'Accuracy',
    value: '87%',
  },
  {
    key: 'autonomy',
    name: 'Autonomy Optimizer',
    icon: ShieldCheck,
    desc: 'AURA-framework risk scoring that raises or lowers how much the agent is allowed to do alone.',
    metric: 'AIx',
    value: '0.75',
  },
  {
    key: 'reasoning',
    name: 'Reasoning & Planning',
    icon: Route,
    desc: 'Agentic planning engine with uncertainty-aware execution and adaptive context compression.',
    metric: 'Success',
    value: '90%',
  },
]

const TASK_TYPES = ['bill_payment', 'appointment', 'paperwork']

function StatusPill() {
  const [online, setOnline] = useState(null)

  useEffect(() => {
    let cancelled = false
    getMetrics()
      .then(() => !cancelled && setOnline(true))
      .catch(() => !cancelled && setOnline(false))
    return () => { cancelled = true }
  }, [])

  return (
    <div className="navbar-status-pill">
      <span className={`status-dot ${online ? 'online' : ''}`} />
      {online === null ? 'checking API…' : online ? BASE_URL.replace(/^https?:\/\//, '') : 'API unreachable'}
    </div>
  )
}

function ModuleGrid() {
  return (
    <div className="module-grid">
      {MODULES.map((m) => {
        const Icon = m.icon
        return (
          <div className="glass-card module-card" key={m.key}>
            <div className="module-card-header">
              <div className="module-icon"><Icon size={18} /></div>
            </div>
            <div className="module-name">{m.name}</div>
            <div className="module-desc">{m.desc}</div>
            <div className="module-metric-row">
              <span>{m.metric}</span>
              <span>{m.value}</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function TaskConsole() {
  const [taskType, setTaskType] = useState(TASK_TYPES[0])
  const [userId, setUserId] = useState('U001')
  const [inputData, setInputData] = useState('{\n  "amount": 150,\n  "payee": "Electric Co",\n  "due": "2026-09-20"\n}')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const parsed = JSON.parse(inputData)
      const res = await createTask({ task_type: taskType, user_id: userId, input_data: parsed })
      setResult(res)
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  const statusIcon = {
    completed: <CheckCircle2 size={14} />,
    pending_intervention: <AlertTriangle size={14} />,
    failed: <XCircle size={14} />,
  }

  return (
    <div className="console-grid">
      <form className="glass-card console-panel" onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Task Type</label>
          <select className="form-select" value={taskType} onChange={(e) => setTaskType(e.target.value)}>
            {TASK_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label>User ID</label>
          <input className="form-input" value={userId} onChange={(e) => setUserId(e.target.value)} />
        </div>
        <div className="form-group">
          <label>Input Data (JSON)</label>
          <textarea className="form-textarea" value={inputData} onChange={(e) => setInputData(e.target.value)} />
        </div>
        <button className="btn-primary" type="submit" disabled={loading}>
          {loading ? <Loader2 size={16} className="spin" /> : <Send size={16} />}
          {loading ? ' Dispatching…' : ' Run Task'}
        </button>
      </form>

      <div className="glass-card result-panel">
        {error && <div className="error-box">{error}</div>}
        {!error && !result && (
          <div className="result-empty">Submit a task to see the orchestrator's structured output here.</div>
        )}
        {result && (
          <>
            <div className="result-row">
              <span className="result-label">Task ID</span>
              <span className="result-value">{result.task_id?.slice(0, 8)}…</span>
            </div>
            <div className="result-row">
              <span className="result-label">Status</span>
              <span className={`badge ${result.status}`}>
                {statusIcon[result.status]} {result.status}
              </span>
            </div>
            <div className="result-row">
              <span className="result-label">Intervention Required</span>
              <span className="result-value">{String(result.intervention_required)}</span>
            </div>
            <div className="result-row">
              <span className="result-label">Execution Time</span>
              <span className="result-value">{(result.execution_time * 1000).toFixed(1)} ms</span>
            </div>
            <div className="result-row" style={{ display: 'block' }}>
              <span className="result-label">Output Data</span>
              <pre className="mono" style={{ marginTop: 8, fontSize: 12, color: 'var(--primary)', whiteSpace: 'pre-wrap' }}>
                {JSON.stringify(result.output_data, null, 2)}
              </pre>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function MetricsPanel() {
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getMetrics().then(setMetrics).catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="error-box">Could not load metrics — {error}</div>
  if (!metrics) return <div className="result-empty">Loading metrics…</div>

  const cards = [
    { label: 'Total Tasks', value: metrics.total_tasks },
    { label: 'Successful', value: metrics.successful_tasks },
    { label: 'Success Rate', value: `${(metrics.success_rate * 100).toFixed(0)}%` },
    { label: 'Avg Exec Time', value: `${(metrics.avg_execution_time * 1000).toFixed(0)} ms` },
  ]

  return (
    <div className="metrics-grid">
      {cards.map((c) => (
        <div className="glass-card metric-card" key={c.label}>
          <div className="metric-value">{c.value}</div>
          <div className="metric-label">{c.label}</div>
        </div>
      ))}
    </div>
  )
}

export default function App() {
  return (
    <div className="app">
      <nav className="navbar">
        <div className="navbar-logo-pill">
          <Activity size={18} /> AIAA
        </div>
        <StatusPill />
      </nav>

      <header className="hero">
        <div className="hero-eyebrow">Autonomous Intervention-Aware Agent</div>
        <h1>Autonomous AI that knows when to ask for help.</h1>
        <p>
          Five research-backed modules — memory, intervention timing, style
          adaptation, autonomy control, and reasoning — working together
          behind one orchestrator, one task at a time.
        </p>
      </header>

      <section className="section">
        <h2 className="section-title"><span className="accent-bar" />Modules</h2>
        <ModuleGrid />
      </section>

      <section className="section">
        <h2 className="section-title"><span className="accent-bar" />Run a Task</h2>
        <TaskConsole />
      </section>

      <section className="section">
        <h2 className="section-title"><span className="accent-bar" />Live Metrics</h2>
        <MetricsPanel />
      </section>

      <footer className="footer">
        AIAA · connected to <a href={BASE_URL}>{BASE_URL}</a>
      </footer>
    </div>
  )
}
