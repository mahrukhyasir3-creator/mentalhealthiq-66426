import { useState } from 'react'
import { predictAndSave } from '../lib/api'

const initialDPQ = Array.from({ length: 9 }).map(() => 0)

export default function PredictionForm() {
  const [age, setAge] = useState(30)
  const [gender, setGender] = useState(1)
  const [race, setRace] = useState(1)
  const [income, setIncome] = useState(2)
  const [education, setEducation] = useState(4)
  const [marital, setMarital] = useState(1)
  const [ageGroup, setAgeGroup] = useState('26-40')
  const [dpq, setDpq] = useState<number[]>(initialDPQ)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const onDpqChange = (idx: number, value: number) => {
    const copy = [...dpq]
    copy[idx] = value
    setDpq(copy)
  }

  const submit = async (e: any) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    const payload: any = {
      RIDAGEYR: age,
      RIAGENDR: gender,
      RIDRETH1: race,
      INDHHIN2: income,
      DMDEDUC2: education,
      DMDMARTL: marital,
      AGE_GROUP: ageGroup,
    }
    for (let i = 0; i < 9; i++) {
      payload[`DPQ0${i+1}`] = dpq[i]
    }

    try {
      const res = await predictAndSave(payload)
      setResult(res)
    } catch (err: any) {
      setError(err?.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white p-4 rounded shadow-sm">
      <h3 className="text-lg font-semibold mb-4">Predict Depression Severity</h3>
      <form onSubmit={submit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <label className="space-y-1">
            <div className="text-sm text-gray-600">Age</div>
            <input type="number" value={age} onChange={e => setAge(Number(e.target.value))} className="w-full border rounded px-2 py-1" />
          </label>

          <label className="space-y-1">
            <div className="text-sm text-gray-600">Gender (code)</div>
            <input type="number" value={gender} onChange={e => setGender(Number(e.target.value))} className="w-full border rounded px-2 py-1" />
          </label>

          <label className="space-y-1">
            <div className="text-sm text-gray-600">Race (code)</div>
            <input type="number" value={race} onChange={e => setRace(Number(e.target.value))} className="w-full border rounded px-2 py-1" />
          </label>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <label className="space-y-1">
            <div className="text-sm text-gray-600">Income (code)</div>
            <input type="number" value={income} onChange={e => setIncome(Number(e.target.value))} className="w-full border rounded px-2 py-1" />
          </label>
          <label className="space-y-1">
            <div className="text-sm text-gray-600">Education (code)</div>
            <input type="number" value={education} onChange={e => setEducation(Number(e.target.value))} className="w-full border rounded px-2 py-1" />
          </label>
          <label className="space-y-1">
            <div className="text-sm text-gray-600">Marital Status (code)</div>
            <input type="number" value={marital} onChange={e => setMarital(Number(e.target.value))} className="w-full border rounded px-2 py-1" />
          </label>
        </div>

        <div>
          <div className="text-sm text-gray-600 mb-2">Age Group</div>
          <select value={ageGroup} onChange={e => setAgeGroup(e.target.value)} className="w-full border rounded px-2 py-1">
            <option>18-25</option>
            <option>26-40</option>
            <option>41-55</option>
            <option>56-70</option>
            <option>71+</option>
          </select>
        </div>

        <div>
          <div className="text-sm text-gray-600 mb-2">PHQ-9 Items (0-3)</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            {dpq.map((v, i) => (
              <label key={i} className="space-y-1">
                <div className="text-xs text-gray-500">DPQ0{i+1}</div>
                <input type="number" min={0} max={3} value={v} onChange={e => onDpqChange(i, Number(e.target.value))} className="w-full border rounded px-2 py-1" />
              </label>
            ))}
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded" disabled={loading}>{loading ? 'Predicting...' : 'Submit & Save'}</button>
          <div className="text-xs text-gray-500">This prediction is not a medical diagnosis.</div>
        </div>
      </form>

      {error && <div className="mt-4 text-red-600">{error}</div>}

      {result && (
        <div className="mt-4 p-3 border rounded bg-gray-50">
          <div className="font-semibold">Result</div>
          <div>Severity: <strong>{result.severity}</strong></div>
          <div>Risk score: <strong>{result.risk_score}</strong></div>
          <div className="text-xs text-gray-500 mt-2">{result.warning}</div>
        </div>
      )}
    </div>
  )
}
