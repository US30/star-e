/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html",
  ],
  theme: {
    extend: {
      colors: {
        bull: '#22c55e',
        bear: '#ef4444',
        sideways: '#64748b',
      },
    },
  },
  plugins: [],
}
