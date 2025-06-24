/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'ba-blue': '#1f77b4',
        'ba-green': '#28a745',
        'ba-light': '#f0f2f6',
      },
    },
  },
  plugins: [],
} 