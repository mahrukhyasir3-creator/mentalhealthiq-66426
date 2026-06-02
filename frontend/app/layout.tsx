import '../globals.css'
import Navbar from '../../components/Navbar'
import Sidebar from '../../components/Sidebar'

export const metadata = {
  title: 'MentalHealthIQ Dashboard'
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-100 min-h-screen">
        <Navbar />
        <div className="max-w-7xl mx-auto px-4 py-6 flex gap-6">
          <Sidebar />
          <main className="flex-1">{children}</main>
        </div>
      </body>
    </html>
  )
}
