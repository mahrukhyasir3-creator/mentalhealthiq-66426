import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export default function FairnessChart({ data, dataKey, title }: { data: any[]; dataKey: string; title: string }) {
  return (
    <div className="bg-white p-4 rounded shadow-sm h-64">
      <h4 className="text-sm font-semibold mb-2">{title}</h4>
      <ResponsiveContainer width="100%" height="85%">
        <BarChart data={data}>
          <XAxis dataKey="group" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey={dataKey} fill="#3182ce" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
