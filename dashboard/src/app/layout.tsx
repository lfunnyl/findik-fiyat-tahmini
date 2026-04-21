// dashboard/src/app/layout.tsx
import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: '🌰 Fındık Fiyat Tahmin Sistemi',
  description:
    'Türkiye fındık serbest piyasa fiyatı tahmin sistemi. XGBoost + Ridge Weighted Ensemble, ' +
    'Conformal Prediction %90 güven aralığı, Double ML Causal Inference.',
  keywords: ['fındık', 'fiyat tahmini', 'Türkiye', 'machine learning', 'XGBoost'],
  authors: [{ name: 'lfunnyl' }],
  openGraph: {
    title: '🌰 Fındık Fiyat Tahmin Sistemi',
    description: '2026 yılı Türkiye fındık fiyat tahminleri — Weighted Ensemble Model',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="tr">
      <body>{children}</body>
    </html>
  )
}
