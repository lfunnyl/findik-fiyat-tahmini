export type Language = 'tr' | 'en';

export const translations = {
  tr: {
    sidebarTitle: '🌰 Fındık',
    sidebarSubtitle: 'Fiyat Tahmin v3.1',
    usdRate: 'USD/TRY Kuru',
    monthlyRateIncrease: 'Aylık Kur Artışı (%)',
    confidenceInterval: 'Güven Aralığı',
    algorithm: 'Algoritma',
    target: 'Hedef',
    data: 'Veri',
    months: 'aylık',
    lastData: 'Son veri',
    realUsd: 'Reel USD/kg',
    
    heroTitle: 'Fındık Fiyat Tahmin Sistemi',
    heroSub: '2026 yılı aylık serbest piyasa fiyat tahminleri · Conformal Prediction %90 CI · Double ML',
    forecast2026: '2026 Tahmini',
    warning: 'Bu sistem bir karar destek aracıdır. Spekülatif hareketler ve ani jeopolitik olaylar öngörülemez.',
    
    tabMain: '🔮 Ana Tahminler',
    tabModel: '📊 Model Analizi',
    tabWhatIf: '⚡ What-If',

    apr2026: 'Nisan 2026',
    dec2026: 'Aralık 2026',
    avg2026: '2026 Ort.',
    tmoForecast: 'TMO Tahmin',
    marginOfError: 'Hata Payı',
    expectedIncrease: 'beklenen artış',
    
    chartTitle: '📈 2026 Aylık Fiyat Tahmini (TL/kg)',
    actualPrice: 'Gerçek Fiyat',
    ci90: '%90 Güven Aralığı',
    harvest: 'Hasat',

    tableTitle: '📋 Aylık Tahmin Tablosu',
    thMonth: 'Ay',
    thUsdTry: 'USD/TRY',
    thNominalUsd: 'Nominal USD/kg',
    thForecastTl: 'TL/kg Tahmin',
    thLowCi: 'Alt CI',
    thHighCi: 'Üst CI',

    causalEffectAte: 'Nedensel Etki (ATE)',
    causalEffectDesc: '1 birim kur artışı → fiyata etki',
    modelComparison: '📊 Model Karşılaştırması',
    doubleMlEffect: '🔬 Double ML Nedensel Etki',
    modelFactors: '🧬 Model Karar Faktörleri (SHAP)',
    modelFactorsDesc: 'Modelin fiyatı tahmin ederken hangi değişkenlere ne kadar güvendiğini gösterir.',
    importanceScore: 'Önem Skoru',

    forexFinance: '💱 Döviz & Finans',
    brentOil: 'Brent Petrol',
    harvestYield: '🌾 Rekolte',
    change: 'Değişim',
    shortage: '🔴 Büyük kıtlık → fiyat baskısı yüksek',
    bountiful: '🟢 Bol hasat → fiyat baskısı düşük',
    normalSeason: '⚪ Normal sezon',
    scenarioResult: '🎯 Senaryo Sonucu',
    baseScenario: 'Baz senaryo',

    footer: '🌰 Fındık Fiyat Tahmin Sistemi v3.1 · XGBoost + Ridge Weighted Ensemble · Conformal Prediction · Double ML · 2026'
  },
  en: {
    sidebarTitle: '🌰 Hazelnut',
    sidebarSubtitle: 'Price Forecast v3.1',
    usdRate: 'USD/TRY Rate',
    monthlyRateIncrease: 'Monthly Exchange Rate Increase (%)',
    confidenceInterval: 'Confidence Interval',
    algorithm: 'Algorithm',
    target: 'Target',
    data: 'Data',
    months: 'months',
    lastData: 'Last data',
    realUsd: 'Real USD/kg',
    
    heroTitle: 'Hazelnut Price Forecast System',
    heroSub: '2026 monthly free market price forecasts · Conformal Prediction 90% CI · Double ML',
    forecast2026: '2026 Forecast',
    warning: 'This system is a decision support tool. Speculative movements and sudden geopolitical events cannot be predicted.',
    
    tabMain: '🔮 Main Forecasts',
    tabModel: '📊 Model Analysis',
    tabWhatIf: '⚡ What-If',

    apr2026: 'April 2026',
    dec2026: 'December 2026',
    avg2026: '2026 Avg.',
    tmoForecast: 'TMO Forecast',
    marginOfError: 'Margin of Error',
    expectedIncrease: 'expected increase',
    
    chartTitle: '📈 2026 Monthly Price Forecast (TRY/kg)',
    actualPrice: 'Actual Price',
    ci90: '90% Confidence Interval',
    harvest: 'Harvest',

    tableTitle: '📋 Monthly Forecast Table',
    thMonth: 'Month',
    thUsdTry: 'USD/TRY',
    thNominalUsd: 'Nominal USD/kg',
    thForecastTl: 'Forecast TRY/kg',
    thLowCi: 'Lower CI',
    thHighCi: 'Upper CI',

    causalEffectAte: 'Causal Effect (ATE)',
    causalEffectDesc: '1 unit rate increase → effect on price',
    modelComparison: '📊 Model Comparison',
    doubleMlEffect: '🔬 Double ML Causal Effect',
    modelFactors: '🧬 Model Decision Factors (SHAP)',
    modelFactorsDesc: 'Shows how much the model relies on which variables when predicting the price.',
    importanceScore: 'Importance Score',

    forexFinance: '💱 Forex & Finance',
    brentOil: 'Brent Crude Oil',
    harvestYield: '🌾 Harvest Yield',
    change: 'Change',
    shortage: '🔴 Severe shortage → high price pressure',
    bountiful: '🟢 Bountiful harvest → low price pressure',
    normalSeason: '⚪ Normal season',
    scenarioResult: '🎯 Scenario Result',
    baseScenario: 'Base scenario',

    footer: '🌰 Hazelnut Price Forecast System v3.1 · XGBoost + Ridge Weighted Ensemble · Conformal Prediction · Double ML · 2026'
  }
};
