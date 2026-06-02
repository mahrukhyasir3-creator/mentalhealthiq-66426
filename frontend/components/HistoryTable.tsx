export default function HistoryTable({ rows }: { rows: any[] }) {
  return (
    <div className="bg-white p-4 rounded shadow-sm overflow-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-gray-500">
          <tr>
            <th className="p-2">Timestamp</th>
            <th className="p-2">Age</th>
            <th className="p-2">Gender</th>
            <th className="p-2">Severity</th>
            <th className="p-2">Risk</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, idx) => (
            <tr key={idx} className="border-t">
              <td className="p-2">{new Date(r.timestamp).toLocaleString()}</td>
              <td className="p-2">{r.input_data?.RIDAGEYR ?? '-'}</td>
              <td className="p-2">{r.input_data?.RIAGENDR ?? '-'}</td>
              <td className="p-2">{r.severity}</td>
              <td className="p-2">{r.risk_score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
