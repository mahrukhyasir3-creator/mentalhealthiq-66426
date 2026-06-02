import Link from 'next/link'

export default function Sidebar() {
  return (
    <aside className="w-64 bg-gray-50 border-r hidden md:block">
      <div className="p-4">
        <h3 className="text-sm font-semibold mb-4">Navigation</h3>
        <ul className="space-y-2 text-sm">
          <li><Link href="/dashboard" className="block p-2 rounded hover:bg-gray-100">Dashboard</Link></li>
          <li><Link href="/predict" className="block p-2 rounded hover:bg-gray-100">Predict</Link></li>
          <li><Link href="/fairness" className="block p-2 rounded hover:bg-gray-100">Fairness</Link></li>
          <li><Link href="/history" className="block p-2 rounded hover:bg-gray-100">History</Link></li>
          <li><Link href="/metrics" className="block p-2 rounded hover:bg-gray-100">Metrics</Link></li>
        </ul>
      </div>
    </aside>
  )
}
