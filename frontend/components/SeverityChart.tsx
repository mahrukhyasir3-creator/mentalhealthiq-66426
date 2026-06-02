import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts'

const COLORS = ['#2b6cb0', '#3182ce', '#63b3ed', '#f6ad55', '#e53e3e']

export default function SeverityChart({ data }: { data: { name: string; value: number }[] }) {
  return (
    <div className="bg-white p-4 rounded shadow-sm h-64">
      <h4 className="text-sm font-semibold mb-2">Severity Distribution</h4>
      <ResponsiveContainer width="100%" height="85%">
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" innerRadius={40} outerRadius={80} label>
            {data.map((_, idx) => (
              <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}
