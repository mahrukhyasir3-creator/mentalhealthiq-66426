import PredictionForm from '../../components/PredictionForm'

export default function PredictPage() {
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Predict</h2>
      <div className="max-w-3xl">
        <PredictionForm />
        <div className="mt-4 text-xs text-gray-500">Medical disclaimer: This tool provides an automated estimate and is not a substitute for professional diagnosis or treatment.</div>
      </div>
    </div>
  )
}
