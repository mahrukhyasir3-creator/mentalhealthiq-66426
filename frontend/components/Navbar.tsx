import Link from 'next/link'

export default function Navbar() {
  return (
    <header className="w-full bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <Link href="/" className="text-lg font-semibold">MentalHealthIQ</Link>
        <nav className="space-x-4">
          <Link href="/dashboard" className="text-sm text-gray-600">Dashboard</Link>
          <Link href="/predict" className="text-sm text-gray-600">Predict</Link>
          <Link href="/fairness" className="text-sm text-gray-600">Fairness</Link>
          <Link href="/history" className="text-sm text-gray-600">History</Link>
          <Link href="/metrics" className="text-sm text-gray-600">Metrics</Link>
        </nav>
      </div>
    </header>
  )
}
