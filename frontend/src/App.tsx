import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Leaderboard from './pages/Leaderboard'
import ModelDetail from './pages/ModelDetail'
import Compare from './pages/Compare'
import About from './pages/About'

export default function App() {
  return (
    <div className="min-h-screen bg-dm-bg">
      <Header />
      <main>
        <Routes>
          <Route path="/" element={<Leaderboard />} />
          <Route path="/model/:slug" element={<ModelDetail />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
      <footer className="border-t border-dm-border py-8 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-dm-muted text-sm">
            StabilityBench &middot; LLM Recursive Stability (RSI) Leaderboard &middot; {new Date().getFullYear()}
          </p>
          <p className="text-dm-border text-xs mt-1">
            Based on the Constraint Residual Framework. RSI = recursive stability index per generation.
          </p>
        </div>
      </footer>
    </div>
  )
}
