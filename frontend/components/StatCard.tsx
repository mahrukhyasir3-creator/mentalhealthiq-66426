type Props = { title: string; value: string | number; subtitle?: string }

export default function StatCard({ title, value, subtitle }: Props) {
  return (
    <div className="bg-white p-4 rounded shadow-sm">
      <div className="text-xs text-gray-500">{title}</div>
      <div className="mt-2 text-2xl font-bold">{value}</div>
      {subtitle && <div className="text-sm text-gray-400 mt-1">{subtitle}</div>}
    </div>
  )
}
