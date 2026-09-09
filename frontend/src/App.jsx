import { useEffect, useMemo, useState } from 'react'

const percent = (value) => `${((value || 0) * 100).toFixed(2)}%`

function Metric({ label, value, accent = false }) {
  return <article className={`metric ${accent ? 'accent' : ''}`}><span>{label}</span><strong>{value}</strong></article>
}

function LineChart({ history, fields, title, percentage = false }) {
  const width = 680
  const height = 230
  const padding = 30
  const values = fields.flatMap((field) => history.map((row) => row[field.key] || 0))
  const maximum = Math.max(...values, 0.01)
  const minimum = Math.min(...values, 0)
  const x = (index) => padding + (index * (width - padding * 2)) / Math.max(history.length - 1, 1)
  const y = (value) => height - padding - ((value - minimum) / Math.max(maximum - minimum, 0.01)) * (height - padding * 2)
  const pathFor = (key) => history.map((row, index) => `${index ? 'L' : 'M'} ${x(index)} ${y(row[key] || 0)}`).join(' ')

  return <section className="chart-card">
    <div className="chart-heading"><h3>{title}</h3><div>{fields.map((field) => <span className="legend" key={field.key}><i style={{ background: field.color }} />{field.label}</span>)}</div></div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
      {[0, 0.5, 1].map((tick) => <line key={tick} x1={padding} x2={width - padding} y1={padding + tick * (height - padding * 2)} y2={padding + tick * (height - padding * 2)} className="grid" />)}
      {fields.map((field) => <path key={field.key} d={pathFor(field.key)} fill="none" stroke={field.color} strokeWidth="3" strokeLinecap="round" />)}
      {history.map((row, index) => <text className="axis" key={row.epoch} x={x(index)} y={height - 8} textAnchor="middle">E{row.epoch}</text>)}
    </svg>
    <p className="chart-value">Latest: {percentage ? percent(history.at(-1)?.[fields.at(-1).key]) : history.at(-1)?.[fields.at(-1).key]?.toFixed(3)}</p>
  </section>
}

export default function App() {
  const [runs, setRuns] = useState([])
  const [selected, setSelected] = useState('')
  const [detail, setDetail] = useState(null)
  const [error, setError] = useState('')
  const loadRuns = async () => {
    try {
      const response = await fetch('/api/runs')
      if (!response.ok) throw new Error('Không thể kết nối Training API.')
      const payload = await response.json()
      setRuns(payload)
      setSelected((current) => current || payload[0]?.name || '')
      setError('')
    } catch (caught) { setError(caught.message) }
  }
  useEffect(() => { loadRuns() }, [])
  useEffect(() => {
    if (!selected) return
    fetch(`/api/runs/${selected}`).then((response) => response.json()).then(setDetail).catch(() => setError('Không thể đọc chi tiết run.'))
  }, [selected])
  const history = detail?.history || []
  const best = detail?.status?.best_validation_top1 ?? detail?.summary?.best_validation_top1 ?? 0
  const latest = useMemo(() => history.at(-1), [history])

  return <main className="shell">
    <header><div><p className="eyebrow">LOCAL EXPERIMENT TRACKING</p><h1>Food-101 Training <em>Dashboard</em></h1><p className="subhead">ResNet experiments, validation curves and official test results.</p></div><button onClick={loadRuns}>↻ Refresh data</button></header>
    {error && <p className="error">{error} Hãy chạy FastAPI ở port 8000 trước.</p>}
    <section className="run-switcher"><label htmlFor="run">Training run</label><select id="run" value={selected} onChange={(event) => setSelected(event.target.value)}>{runs.map((run) => <option key={run.name} value={run.name}>{run.name}</option>)}</select><span className={`pill ${detail?.status?.state || ''}`}>{detail?.status?.state || 'loading'}</span></section>
    {!detail ? <section className="empty">Loading experiment data…</section> : <>
      <section className="metrics"><Metric label="Best validation Top-1" value={percent(best)} accent /><Metric label="Latest train Top-1" value={percent(latest?.train_top1)} /><Metric label="Epoch" value={`${detail.status.current_epoch || 0} / ${detail.status.total_epochs || '?'}`} /><Metric label="Device" value={detail.status.device || detail.summary.device || 'unknown'} /></section>
      {history.length ? <section className="charts"><LineChart title="Training vs validation loss" history={history} fields={[{ key: 'train_loss', label: 'Train', color: '#ff725e' }, { key: 'validation_loss', label: 'Validation', color: '#4e6cff' }]} /><LineChart title="Training vs validation Top-1" history={history} percentage fields={[{ key: 'train_top1', label: 'Train', color: '#ffb443' }, { key: 'validation_top1', label: 'Validation', color: '#20b58c' }]} /></section> : <section className="empty">Metrics sẽ hiện sau epoch đầu tiên.</section>}
      {detail.evaluation && <section className="evaluation"><div><p className="eyebrow">OFFICIAL TEST SPLIT</p><h2>Evaluation results</h2><div className="test-metrics"><Metric label="Top-1" value={percent(detail.evaluation.top1_accuracy)} /><Metric label="Top-5" value={percent(detail.evaluation.top5_accuracy)} /><Metric label="Macro F1" value={percent(detail.evaluation.macro_f1)} /></div></div>{detail.has_confusion_matrix && <img src={`/api/runs/${detail.name}/confusion-matrix`} alt="Confusion matrix" />}</section>}
    </>}
  </main>
}
