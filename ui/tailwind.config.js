/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#090a0f',
        surface: '#11131c',
        surfaceLight: '#181b29',
        borderDark: '#23283a',
        accentCyan: '#06b6d4',
        accentBlue: '#3b82f6',
        accentViolet: '#8b5cf6',
        accentEmerald: '#10b981',
      }
    },
  },
  plugins: [],
}

