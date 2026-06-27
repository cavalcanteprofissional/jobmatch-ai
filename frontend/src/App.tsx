import { Routes, Route, Navigate } from 'react-router-dom'
import JobMatch from './pages/JobMatch'
import Monitor from './pages/Monitor'
import { useTheme } from './context/ThemeContext'

function App() {
  const { theme, toggle } = useTheme()

  return (
    <div className="min-h-screen bg-primary dark:bg-primary-dark transition-colors">
      <nav className="bg-white dark:bg-gray-900 shadow-sm border-b border-gray-200 dark:border-gray-700 transition-colors">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-6">
          <a href="/" className="font-bold text-lg text-indigo-600 dark:text-indigo-400">JobMatch AI</a>
          <a href="/" className="text-sm text-gray-600 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">Match</a>
          <a href="/monitor" className="text-sm text-gray-600 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">Monitor</a>
          <div className="ml-auto">
            <button
              onClick={toggle}
              className="p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? '\u2600\uFE0F' : '\uD83C\uDF19'}
            </button>
          </div>
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
