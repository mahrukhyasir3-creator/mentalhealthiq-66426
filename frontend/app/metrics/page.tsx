import { useEffect, useState } from 'react'
import { getMetrics } from '../../lib/api'

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<any | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    ;(async () => {
      const data = await getMetrics()
      setMetrics(data)
      setLoading(false)
    })()
  }, [])

  if (loading) return <div>Loading...</div>

  if (!metrics) {
    return (
      <div>
        <h2 className="text-xl font-semibold">Metrics</h2>
        <div className="bg-white p-4 rounded shadow-sm">API Metrics not available. Showing mock UI.</div>
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Metrics</h2>
      <div className="bg-white p-4 rounded shadow-sm">
        <pre className="text-sm">{JSON.stringify(metrics.metrics, null, 2)}</pre>
      </div>
    </div>
  )
}
