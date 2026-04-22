'use client'
import React, { useState } from 'react'
import useSWR from 'swr'
import { api, MonthlyPrediction, HistoricalPrice } from '@/lib/api'
import dynamic from 'next/dynamic'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

const DARK = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Inter', color: 'rgba(255,255,255,0.8)', size: 12 }, margin: { l: 10, r: 10, t: 20, b: 10 } }

function MetricCard({ label, value, delta, deltaNeg }: { label: string; value: string; delta?: string; deltaNeg?: boolean }) {
  return (
    <div className="metric-card fade-up">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {delta && <div className={`metric-delta ${deltaNeg ? 'negative' : ''}`}>{delta}</div>}
    </div>
  )
}

function Skeleton({ h = 200 }: { h?: number }) {
  return <div className="skeleton" style={{ height: h }} />
}

export default function Home() {
  const [tab, setTab] = useState<'tahmin' | 'model' | 'whatif'>('tahmin')
  const [kur, setKur] = useState(44)
  const [kurArtis, setKurArtis] = useState(0.8)
  const [showCI, setShowCI] = useState(true)
  // what-if state
  const [wiKur, setWiKur] = useState(44)
  const [wiRekolte, setWiRekolte] = useState(0)
  const [wiPetrol, setWiPetrol] = useState(85)

  const { data: pred, isLoading: predLoad } = useSWR(
    ['predict', kur, kurArtis],
    () => api.predict(kur, kurArtis / 100),
    { revalidateOnFocus: false }
  )
  const { data: scores } = useSWR('scores', api.scores, { revalidateOnFocus: false })
  const { data: tmo } = useSWR('tmo', api.tmo, { revalidateOnFocus: false })
  const { data: causal } = useSWR('causal', api.causal, { revalidateOnFocus: false })
  const { data: history } = useSWR('history', () => api.history(30), { revalidateOnFocus: false })
  const { data: info } = useSWR('info', api.info, { revalidateOnFocus: false })
  const { data: shap } = useSWR('shap', api.shap, { revalidateOnFocus: false })
  const { data: wi } = useSWR(
    ['whatif', wiKur, wiRekolte, wiPetrol],
    () => api.whatif({ usd_try: wiKur, brent_petrol: wiPetrol, rekolte_degisim_pct: wiRekolte, ihracat_degisim_pct: 0 }),
    { revalidateOnFocus: false }
  )

  const preds = pred?.predictions ?? []
  const nisTL = preds[0]?.tl ?? 0
  const aralikTL = preds[preds.length - 1]?.tl ?? 0
  const ortTL = preds.length ? preds.reduce((s: number, p: MonthlyPrediction) => s + p.tl, 0) / preds.length : 0

  const predDates = preds.map((p: MonthlyPrediction) => `2026-${String(p.ay).padStart(2, '0')}-01`)

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div style={{ marginBottom: 28 }}>
          <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>🌰 Fındık</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>Fiyat Tahmin v3.1</div>
        </div>

        <div style={{ marginBottom: 24 }}>
          <div className="metric-label" style={{ marginBottom: 8 }}>USD/TRY Kuru</div>
          <input type="range" min={30} max={60} step={0.5} value={kur} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setKur(+e.target.value)} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: 4 }}>
            <span>30</span><span style={{ color: 'var(--accent)', fontWeight: 600 }}>{kur} ₺</span><span>60</span>
          </div>
        </div>

        <div style={{ marginBottom: 24 }}>
          <div className="metric-label" style={{ marginBottom: 8 }}>Aylık Kur Artışı (%)</div>
          <input type="range" min={-2} max={5} step={0.1} value={kurArtis} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setKurArtis(+e.target.value)} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: 4 }}>
            <span>-2%</span><span style={{ color: 'var(--accent)', fontWeight: 600 }}>%{kurArtis}</span><span>5%</span>
          </div>
        </div>

        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.85rem', cursor: 'pointer', marginBottom: 16 }}>
          <input type="checkbox" checked={showCI} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setShowCI(e.target.checked)} />
          Güven Aralığı
        </label>

        <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '20px 0' }} />

        <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
          <div><strong>Algoritma:</strong> {pred?.model_info?.algorithm ?? info?.model ?? 'Weighted Ensemble'}</div>
          {pred?.model_info?.weights && (
            <div>
              {Object.entries(pred.model_info.weights)
                .filter(([, w]) => (w as number) > 0.01)
                .map(([name, w]) => (
                  <span key={name} style={{ color: 'var(--accent)' }}>{name} %{Math.round((w as number) * 100)} </span>
                ))
              }
            </div>
          )}
          <div style={{ marginTop: 8 }}><strong>Test MAPE:</strong> %{pred?.model_info?.test_mape?.toFixed(2) ?? '9.05'}</div>
          <div><strong>Test R²:</strong> {pred?.model_info?.test_r2?.toFixed(3) ?? '0.453'}</div>
          <div style={{ marginTop: 8 }}><strong>Hedef:</strong> Reel USD/kg</div>
          <div><strong>Veri:</strong> {info ? `${info.data_rows} aylık` : '152 aylık'}</div>
          {info?.data_end && <div><strong>Son veri:</strong> {info.data_end.slice(0, 7)}</div>}
        </div>
      </aside>

      {/* Main */}
      <main className="main-content">
        {/* Hero */}
        <div className="hero fade-up">
          <div style={{ display: 'flex', gap: 10, marginBottom: 14, flexWrap: 'wrap' }}>
            <span className="badge badge-purple">Weighted Ensemble</span>
            <span className="badge badge-green">MAPE %9.05</span>
            <span className="badge badge-orange">2026 Tahmini</span>
          </div>
          <h1 className="hero-title gradient-text">Fındık Fiyat Tahmin Sistemi</h1>
          <p className="hero-sub">2026 yılı aylık serbest piyasa fiyat tahminleri · Conformal Prediction %90 CI · Double ML</p>
        </div>

        <div className="warning-banner">
          ⚠️ Bu sistem bir <strong>karar destek aracıdır</strong>. Spekülatif hareketler ve ani jeopolitik olaylar öngörülemez.
        </div>

        {/* Tabs */}
        <div className="tab-nav">
          {(['tahmin', 'model', 'whatif'] as const).map(t => (
            <button key={t} className={`tab-btn ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
              {t === 'tahmin' ? '🔮 Ana Tahminler' : t === 'model' ? '📊 Model Analizi' : '⚡ What-If'}
            </button>
          ))}
        </div>

        {/* TAB 1 — Ana Tahminler */}
        {tab === 'tahmin' && (
          <div className="fade-up">
            <div className="grid-5" style={{ marginBottom: 28 }}>
              {predLoad ? Array(5).fill(0).map((_, i) => <Skeleton key={i} h={110} />) : (
                <>
                  <MetricCard label="Nisan 2026" value={`${nisTL.toFixed(1)} TL/kg`} />
                  <MetricCard label="Aralık 2026" value={`${aralikTL.toFixed(1)} TL/kg`} delta={`${(aralikTL - nisTL) > 0 ? '+' : ''}${(aralikTL - nisTL).toFixed(1)} TL`} deltaNeg={aralikTL < nisTL} />
                  <MetricCard label="2026 Ort." value={`${ortTL.toFixed(1)} TL/kg`} />
                  <MetricCard label="TMO Tahmin" value={tmo ? `${tmo.pred_2026.toFixed(1)} TL/kg` : '…'} delta={tmo ? `%${tmo.buyume_pct} beklenen artış` : ''} />
                  <MetricCard label="Hata Payı" value={`%${pred?.model_info?.test_mape?.toFixed(1) ?? '9.1'}`} delta="MAPE (Test)" />
                </>
              )}
            </div>

            {/* Ana Grafik */}
            <div className="card" style={{ marginBottom: 24 }}>
              <div className="section-title">📈 2026 Aylık Fiyat Tahmini (TL/kg)</div>
              {predLoad ? <Skeleton h={380} /> : (
                <Plot
                  data={[
                    ...(history ? [{
                      x: history.map((h: HistoricalPrice) => h.Tarih),
                      y: history.map((h: HistoricalPrice) => h.Serbest_Piyasa_TL_kg),
                      type: 'scatter' as const, mode: 'lines+markers' as const,
                      name: 'Gerçek Fiyat',
                      line: { color: 'rgba(255,255,255,0.5)', width: 2 },
                      marker: { size: 4 },
                    }] : []),
                    ...(showCI ? [{
                      x: [...predDates, ...predDates.slice().reverse()],
                      y: [...preds.map((p: MonthlyPrediction) => p.ci_high), ...preds.map((p: MonthlyPrediction) => p.ci_low).reverse()],
                      fill: 'toself' as const, type: 'scatter' as const,
                      fillcolor: 'rgba(124,106,247,0.1)',
                      line: { color: 'rgba(0,0,0,0)' },
                      name: '%90 Güven Aralığı', hoverinfo: 'skip' as const,
                    }] : []),
                    {
                      x: predDates,
                      y: preds.map((p: MonthlyPrediction) => p.tl),
                      type: 'scatter' as const, mode: 'lines+markers' as const,
                      name: '2026 Tahmin',
                      line: { color: '#7c6af7', width: 3 },
                      marker: { size: 9, color: preds.map((p: MonthlyPrediction) => p.tl), colorscale: 'Viridis', line: { color: 'white', width: 1.5 } },
                      text: preds.map((p: MonthlyPrediction) => `${p.ay_adi}: ${p.tl.toFixed(1)} TL<br>USD: $${p.nominal_usd.toFixed(2)}`),
                      hovertemplate: '%{text}<extra></extra>',
                    },
                  ]}
                  layout={{
                    ...DARK,
                    height: 380,
                    hovermode: 'x unified',
                    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zeroline: false },
                    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zeroline: false, ticksuffix: ' TL' },
                    legend: { bgcolor: 'rgba(255,255,255,0.05)', bordercolor: 'rgba(255,255,255,0.1)', borderwidth: 1 },
                    shapes: [{ type: 'rect', x0: '2026-08-01', x1: '2026-10-15', yref: 'paper', y0: 0, y1: 1, fillcolor: 'rgba(255,152,0,0.06)', line: { width: 0 } }],
                    annotations: [{ x: '2026-09-07', y: 1, yref: 'paper', text: 'Hasat', showarrow: false, font: { color: 'rgba(255,152,0,0.7)', size: 11 } }],
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                  className="plotly-container"
                />
              )}
            </div>

            {/* Tahmin Tablosu */}
            <div className="card">
              <div className="section-title">📋 Aylık Tahmin Tablosu</div>
              {predLoad ? <Skeleton h={300} /> : (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                      {['Ay', 'USD/TRY', 'Nominal USD/kg', 'TL/kg Tahmin', 'Alt CI', 'Üst CI'].map(h => (
                        <th key={h} style={{ padding: '10px 12px', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preds.map((p: MonthlyPrediction) => (
                      <tr key={p.ay} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <td style={{ padding: '10px 12px', fontWeight: 600 }}>{p.ay_adi}</td>
                        <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>{p.kur.toFixed(2)}</td>
                        <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>${p.nominal_usd.toFixed(3)}</td>
                        <td style={{ padding: '10px 12px', fontWeight: 700, color: 'var(--accent-light)' }}>{p.tl.toFixed(1)}</td>
                        <td style={{ padding: '10px 12px', color: 'var(--accent-green)', fontSize: '0.82rem' }}>{p.ci_low.toFixed(1)}</td>
                        <td style={{ padding: '10px 12px', color: 'var(--accent-red)', fontSize: '0.82rem' }}>{p.ci_high.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}

        {/* TAB 2 — Model Analizi */}
        {tab === 'model' && (
          <div className="fade-up">
            <div className="grid-4" style={{ marginBottom: 28 }}>
              <MetricCard label="Test MAPE" value={`%${pred?.model_info?.test_mape?.toFixed(2) ?? scores?.['Weighted Ensemble']?.MAPE?.toFixed(2) ?? '9.05'}`} delta="Weighted Ensemble" />
              <MetricCard label="Test R²" value={(pred?.model_info?.test_r2 ?? scores?.['Weighted Ensemble']?.R2 ?? 0.453).toFixed(3)} />
              <MetricCard label="Test MAE" value={`${(scores?.['Weighted Ensemble']?.MAE ?? 0.494).toFixed(3)} USD/kg`} />
              <MetricCard label="Nedensel Etki (ATE)" value={causal ? `+${causal.average_treatment_effect} TL` : '…'} delta="1 birim kur artışı → fiyata etki" />
            </div>

            {scores && (
              <div className="card" style={{ marginBottom: 24 }}>
                <div className="section-title">📊 Model Karşılaştırması</div>
                <Plot
                  data={[{
                    x: Object.values(scores).map((s: any) => s.MAPE),
                    y: Object.keys(scores),
                    type: 'bar', orientation: 'h',
                    marker: { color: Object.values(scores).map((s: any) => s.MAPE === Math.min(...Object.values(scores).map((x: any) => x.MAPE)) ? '#7c6af7' : '#aec7e8') },
                    text: Object.values(scores).map((s: any) => `${s.MAPE.toFixed(2)}%`),
                    textposition: 'outside',
                  }]}
                  layout={{
                    ...DARK, height: 320, margin: { l: 140, r: 60, t: 10, b: 30 },
                    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zeroline: false, ticksuffix: '%' },
                    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zeroline: false },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
            )}

            {causal && (
              <div className="card glass" style={{ marginBottom: 24 }}>
                <div className="section-title">🔬 Double ML Nedensel Etki</div>
                <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginTop: 12 }}>
                  <div><span style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent)' }}>+{causal.average_treatment_effect} TL/kg</span></div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', maxWidth: 500, lineHeight: 1.7 }}>{causal.yorum}</div>
                </div>
              </div>
            )}

            {shap && (
              <div className="card">
                <div className="section-title">🧬 Model Karar Faktörleri (SHAP)</div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 16 }}>Modelin fiyatı tahmin ederken hangi değişkenlere ne kadar güvendiğini gösterir.</p>
                <Plot
                  data={[{
                    x: shap.map(s => s.importance),
                    y: shap.map(s => s.feature),
                    type: 'bar', orientation: 'h',
                    marker: { 
                      color: shap.map((_, i) => `rgba(124, 106, 247, ${1 - i * 0.05})`),
                      line: { color: 'white', width: 0.5 }
                    }
                  }]}
                  layout={{
                    ...DARK, height: 400, margin: { l: 160, r: 40, t: 10, b: 40 },
                    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zeroline: false, title: { text: 'Önem Skoru' } },
                    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zeroline: false },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
            )}
          </div>
        )}

        {/* TAB 3 — What-If */}
        {tab === 'whatif' && (
          <div className="fade-up">
            <div className="grid-3" style={{ marginBottom: 28 }}>
              <div className="card">
                <div className="section-title">💱 Döviz & Finans</div>
                <label style={{ display: 'block', marginBottom: 8, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>USD/TRY Kuru: <strong style={{ color: 'var(--accent)' }}>{wiKur}</strong></label>
                <input type="range" min={30} max={70} step={0.5} value={wiKur} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setWiKur(+e.target.value)} />
                <label style={{ display: 'block', marginTop: 16, marginBottom: 8, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Brent Petrol: <strong style={{ color: 'var(--accent)' }}>${wiPetrol}</strong></label>
                <input type="range" min={50} max={150} step={1} value={wiPetrol} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setWiPetrol(+e.target.value)} />
              </div>
              <div className="card">
                <div className="section-title">🌾 Rekolte</div>
                <label style={{ display: 'block', marginBottom: 8, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Değişim: <strong style={{ color: wiRekolte < 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>{wiRekolte > 0 ? '+' : ''}{wiRekolte}%</strong></label>
                <input type="range" min={-50} max={50} step={5} value={wiRekolte} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setWiRekolte(+e.target.value)} />
                <div style={{ marginTop: 12, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {wiRekolte < -20 ? '🔴 Büyük kıtlık → fiyat baskısı yüksek' : wiRekolte > 20 ? '🟢 Bol hasat → fiyat baskısı düşük' : '⚪ Normal sezon'}
                </div>
              </div>
              <div className="card">
                <div className="section-title">🎯 Senaryo Sonucu</div>
                {wi ? (
                  <>
                    <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--accent-light)', marginBottom: 8 }}>{wi.whatif_tl.toFixed(1)} TL</div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Baz senaryo: {wi.baz_tl.toFixed(1)} TL</div>
                    <div style={{ fontSize: '0.9rem', fontWeight: 600, color: wi.delta_tl > 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                      {wi.delta_tl > 0 ? '+' : ''}{wi.delta_tl.toFixed(1)} TL ({wi.delta_pct > 0 ? '+' : ''}{wi.delta_pct.toFixed(1)}%)
                    </div>
                  </>
                ) : <Skeleton h={100} />}
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: 48, paddingTop: 20, borderTop: '1px solid var(--border)' }}>
          🌰 Fındık Fiyat Tahmin Sistemi v3.1 · XGBoost + Ridge Weighted Ensemble · Conformal Prediction · Double ML · 2026
        </footer>
      </main>
    </div>
  )
}

