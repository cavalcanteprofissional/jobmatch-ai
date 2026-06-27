/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#ffffff',
          dark: '#111827',
        },
        primary: {
          DEFAULT: '#f9fafb',
          dark: '#030712',
        },
      },
    },
  },
  plugins: [],
}
