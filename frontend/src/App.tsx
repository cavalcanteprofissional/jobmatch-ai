import { Routes, Route, Navigate } from 'react-router-dom'
import JobMatch from './pages/JobMatch'
import Monitor from './pages/Monitor'

function App() {
  return (
    <div className="min-h-screen">
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-6">
          <a href="/" className="font-bold text-lg text-indigo-600">🎯 JobMatch AI</a>
          <a href="/" className="text-sm text-gray-600 hover:text-indigo-600">Match</a>
          <a href="/monitor" className="text-sm text-gray-600 hover:text-indigo-600">Monitor</a>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-4 py-6">
        <Routes>
          <Route path="/" element={<JobMatch />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
