import { useEffect, useState } from 'react'
import { getFairnessReport } from '../../lib/api'
import FairnessChart from '../../components/FairnessChart'

export default function FairnessPage() {
  const [report, setReport] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    ;(async () => {
      try {
        const data = await getFairnessReport()
        setReport(data.records || data)
      } catch (err: any) {
        setError(err?.message || 'Failed to load fairness report')
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  // Group and prepare sample charts for Gender and Income
  const genderData = report
    .filter(r => r.Demographic === 'Gender')
    .map(r => ({ group: r.Gender, Accuracy: Number(r.Accuracy), FPR: Number(r.FPR), FNR: Number(r.FNR), Selection: Number(r.Selection_Rate) }))

  const incomeData = report
    .filter(r => r.Demographic === 'Income')
    .map(r => ({ group: r.Income, Accuracy: Number(r.Accuracy), Selection: Number(r.Selection_Rate) }))

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Fairness</h2>
      {loading && <div>Loading...</div>}
      {error && <div className="text-red-600">{error}</div>}

      {!loading && !error && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FairnessChart data={genderData} dataKey="Accuracy" title="Accuracy by Gender" />
          <FairnessChart data={genderData} dataKey="Selection" title="Selection Rate by Gender" />
          <FairnessChart data={incomeData} dataKey="Accuracy" title="Accuracy by Income" />
          <FairnessChart data={incomeData} dataKey="Selection" title="Selection Rate by Income" />
        </div>
      )}
    </div>
  )
}
