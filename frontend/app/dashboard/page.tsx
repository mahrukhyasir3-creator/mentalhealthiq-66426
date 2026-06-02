import { useEffect, useState } from 'react'
import { getPredictions } from '../../lib/api'
import StatCard from '../../components/StatCard'
import SeverityChart from '../../components/SeverityChart'

export default function DashboardPage() {
  const [rows, setRows] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    ;(async () => {
      setLoading(true)
      try {
        const data = await getPredictions(500)
        setRows(data)
      } catch (err: any) {
        setError(err?.message || 'Failed to load')
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  const total = rows.length
  const avgRisk = (rows.reduce((a, b) => a + (b.risk_score || 0), 0) / Math.max(1, total)).toFixed(3)
  const highRisk = rows.filter(r => (r.risk_score ?? 0) >= 0.8).length

  const severityCounts: Record<string, number> = {}
  rows.forEach(r => {
    const s = r.severity || 'Unknown'
    severityCounts[s] = (severityCounts[s] || 0) + 1
  })
  const severityData = Object.entries(severityCounts).map(([name, value]) => ({ name, value }))

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Dashboard</h2>
      {loading && <div>Loading...</div>}
      {error && <div className="text-red-600">{error}</div>}

      {!loading && !error && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <StatCard title="Total Predictions" value={total} />
          <StatCard title="Average Risk" value={avgRisk} />
          <StatCard title="High Risk (>=0.8)" value={highRisk} />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <SeverityChart data={severityData.length ? severityData : [{ name: 'None', value: 1 }]} />
        <div className="bg-white p-4 rounded shadow-sm h-64">
          <h4 className="text-sm font-semibold mb-2">Predictions per Day</h4>
          <div className="text-sm text-gray-500">(Histogram placeholder) Total: {total}</div>
        </div>
      </div>
    </div>
  )
}
