import Link from 'next/link'

export default function Home() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold">MentalHealthIQ</h1>
      <p className="text-gray-600 mt-2">Open the dashboard to explore predictions and fairness.</p>
      <div className="mt-6 space-x-3">
        <Link href="/dashboard" className="bg-blue-600 text-white px-4 py-2 rounded">Open Dashboard</Link>
        <Link href="/predict" className="bg-white border px-4 py-2 rounded">Make Prediction</Link>
      </div>
    </div>
  )
}
