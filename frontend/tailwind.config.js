/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dm: {
          bg: '#0a0a0f',
          surface: '#141420',
          card: '#1a1a2e',
          border: '#2a2a3e',
          muted: '#6b6b80',
          accent: '#5e5ce6',
          'accent-light': '#7b79f0',
          green: '#30d158',
          blue: '#40a9ff',
          amber: '#faad14',
          red: '#ff4d4f',
        },
      },
      fontFamily: {
        mono: ['SF Mono', 'JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'SF Pro Display', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
