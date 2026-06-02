import { useEffect, useState } from 'react'
import { getPredictions } from '../../lib/api'
import HistoryTable from '../../components/HistoryTable'

export default function HistoryPage() {
  const [rows, setRows] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    ;(async () => {
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

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">History</h2>
      {loading && <div>Loading...</div>}
      {error && <div className="text-red-600">{error}</div>}
      {!loading && !error && <HistoryTable rows={rows} />}
    </div>
  )
}
