/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // API URL'yi build time'da bağla
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  // Plotly için transpile
  transpilePackages: ['react-plotly.js', 'plotly.js'],
}

module.exports = nextConfig
