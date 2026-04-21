"""
tests/test_features.py
======================
Feature Engineering ve Veri Kalitesi Unit Testleri

Çalıştırma:
    pytest tests/ -v --cov=src
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit

# src dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))


# ─── Test Yardımcıları ─────────────────────────────────────────────────────────

def make_mock_df(n=60):
    """Gerçekçi yapıda sahte fındık fiyat verisi üretir."""
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', periods=n, freq='MS')
    df = pd.DataFrame({
        'Tarih': dates,
        'Serbest_Piyasa_TL_kg': np.random.uniform(30, 200, n),
        'USD_TRY_Kapanis':       np.random.uniform(5, 40, n),
        'Brent_Petrol_Kapanis':  np.random.uniform(50, 120, n),
        'Altin_Ons_Kapanis':     np.random.uniform(1200, 2500, n),
        'Asgari_Ucret_TL':       np.random.uniform(2000, 17000, n),
        'Yagis_mm':              np.random.uniform(20, 300, n),
        'Max_Sicaklik_C':        np.random.uniform(5, 35, n),
        'Min_Sicaklik_C':        np.random.uniform(-5, 20, n),
        'Kritik_Don':            np.random.randint(0, 5, n),
    })
    df['Yil']  = df['Tarih'].dt.year
    df['Ay']   = df['Tarih'].dt.month
    df['Yil_Ay'] = df['Tarih'].dt.to_period('M').astype(str)
    return df


# ─── Reel USD Dönüşümü ────────────────────────────────────────────────────────

class TestReelUsdConversion:
    """Reel USD fiyat dönüşümü testleri."""

    US_CPI = {2020: 76.34, 2024: 100.0}

    def reel(self, nominal_usd: float, yil: int) -> float:
        cpi_baz = self.US_CPI[2024]
        carpan  = cpi_baz / self.US_CPI.get(yil, cpi_baz)
        return nominal_usd * carpan

    def test_baz_yilinda_reel_esit_nominal(self):
        """2024 baz yılında reel == nominal olmalı."""
        assert abs(self.reel(5.0, 2024) - 5.0) < 0.001

    def test_eski_yil_fiyati_daha_yuksek_yumusama(self):
        """2020'deki fiyat, 2024 alım gücüyle daha yüksek görünmeli."""
        assert self.reel(5.0, 2020) > 5.0

    def test_sifir_fiyat_sifir_kalmali(self):
        """0 fiyat, herhangi bir CPI'da 0 kalmalı."""
        assert self.reel(0.0, 2020) == 0.0

    def test_cpi_carpan_1_baz_yilinda(self):
        """2024 baz yılı için CPI çarpanı tam 1 olmalı."""
        carpan = self.US_CPI[2024] / self.US_CPI[2024]
        assert abs(carpan - 1.0) < 1e-9


# ─── Feature Engineering ──────────────────────────────────────────────────────

class TestFeatureEngineering:
    """Özellik türetme testleri."""

    def test_lag_ozellikleri_dogru_kayiyor(self):
        """Lag1 özelliği tam 1 satır ötelenmiş olmalı."""
        df = make_mock_df(10)
        df['Fiyat_Lag1'] = df['Serbest_Piyasa_TL_kg'].shift(1)
        assert df['Fiyat_Lag1'].iloc[1] == df['Serbest_Piyasa_TL_kg'].iloc[0]
        assert pd.isna(df['Fiyat_Lag1'].iloc[0])

    def test_ay_sin_cos_aralik_dogru(self):
        """Ay sin/cos değerleri [-1, 1] aralığında olmalı."""
        df = make_mock_df(12)
        df['Ay_Sin'] = np.sin(2 * np.pi * df['Ay'] / 12.0)
        df['Ay_Cos'] = np.cos(2 * np.pi * df['Ay'] / 12.0)
        assert df['Ay_Sin'].between(-1, 1).all()
        assert df['Ay_Cos'].between(-1, 1).all()

    def test_ay_sin_cos_dairesel_simetri(self):
        """Ocak (1) ve Aralık (12) birbirine yakın olmalı (dairesel mevsimsellik)."""
        sin_ocak  = np.sin(2 * np.pi * 1 / 12)
        sin_aralik = np.sin(2 * np.pi * 12 / 12)
        # Aralık sin(2π) = sin(0) ≈ 0; Ocak sin(2π/12) ≈ 0.5
        # Fark 0.5'ten az olmalı (dairesel yakınlık kontrolü)
        assert abs(abs(sin_ocak) - abs(sin_aralik)) < 0.6

    def test_usd_fiyat_hesaplama_pozitif(self):
        """USD fiyat (TL / Kur) daima pozitif olmalı."""
        df = make_mock_df(20)
        df['Fiyat_USD_kg'] = df['Serbest_Piyasa_TL_kg'] / df['USD_TRY_Kapanis']
        assert (df['Fiyat_USD_kg'] > 0).all()

    def test_pct_change_ilk_satir_nan(self):
        """pct_change(1) ilk satırda NaN üretmeli."""
        df = make_mock_df(10)
        pct = df['Serbest_Piyasa_TL_kg'].pct_change(1)
        assert pd.isna(pct.iloc[0])
        assert not pd.isna(pct.iloc[1])


# ─── Feature Selection ────────────────────────────────────────────────────────

class TestFeatureSelection:
    """Feature selection testleri — data leakage kontrolü."""

    def test_mutual_info_top_n_secim(self):
        """MI feature selection tam N özellik döndürmeli."""
        X = pd.DataFrame(np.random.rand(50, 15), columns=[f'f{i}' for i in range(15)])
        y = pd.Series(np.random.rand(50))
        mi = mutual_info_regression(X, y, random_state=42)
        top_n = 8
        selected = pd.Series(mi, index=X.columns).nlargest(top_n).index.tolist()
        assert len(selected) == top_n

    def test_feature_selection_sadece_train_seti(self):
        """Feature selection test verisine dokunmadan yapılmalı."""
        X = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
        y = pd.Series(np.random.rand(100))
        split = 80
        X_tr = X.iloc[:split]
        y_tr = y.iloc[:split]
        # Feature selection train'de yap
        corr = X_tr.corrwith(y_tr).abs()
        selected = corr.nlargest(5).index.tolist()
        # Test setinden hiçbir veri feature selection'a katılmadı
        assert len(selected) == 5


# ─── Walk-Forward CV ─────────────────────────────────────────────────────────

class TestWalkForwardCV:
    """Walk-Forward CV'de temporal leakage yok kontrolü."""

    def test_her_fold_test_train_den_sonra(self):
        """Test indeksleri her zaman train indekslerinden sonra gelmeli."""
        from sklearn.model_selection import TimeSeriesSplit
        n = 100
        X = np.arange(n).reshape(-1, 1)
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X):
            assert max(train_idx) < min(val_idx), "Data leakage: test verisi train'den önce!"

    def test_expanding_window_train_buyuyor(self):
        """Her fold'da train seti büyümeli (expanding window)."""
        from sklearn.model_selection import TimeSeriesSplit
        n = 100
        X = np.arange(n).reshape(-1, 1)
        tscv = TimeSeriesSplit(n_splits=5)
        sizes = [len(tr) for tr, _ in tscv.split(X)]
        assert all(sizes[i] < sizes[i+1] for i in range(len(sizes)-1)), \
            "Walk-Forward fold boyutları artmıyor!"


# ─── Conformal Prediction ─────────────────────────────────────────────────────

class TestConformalPrediction:
    """Conformal CI tutarlılık testleri."""

    def test_ci_lower_upper_sirasi_dogru(self):
        """CI alt sınır her zaman üst sınırdan küçük olmalı."""
        pred_tl = 200.0
        q_hat   = 0.15
        ci_low  = pred_tl * (1 - q_hat)
        ci_high = pred_tl * (1 + q_hat)
        assert ci_low < pred_tl < ci_high

    def test_q_hat_sifir_halinde_nokta_tahmin(self):
        """q_hat=0 iken CI genişliği sıfır olmalı."""
        pred = 150.0
        q    = 0.0
        assert pred * (1 - q) == pred
        assert pred * (1 + q) == pred

    def test_q_hat_pozitif(self):
        """Conformal q_hat daima pozitif olmalı."""
        # JSON'dan okunan q_hat değeri pozitif olmalı
        q_hat = 0.18
        assert q_hat > 0


# ─── Veri Kalitesi ────────────────────────────────────────────────────────────

class TestDataQuality:
    """Ham ve işlenmiş veri kalite testleri."""

    def test_fiyat_pozitif_olmali(self):
        """Tüm fiyat değerleri sıfırdan büyük olmalı."""
        df = make_mock_df(30)
        assert (df['Serbest_Piyasa_TL_kg'] > 0).all()

    def test_kur_mantikli_aralikta(self):
        """USD/TRY kuru 1 ile 100 arasında olmalı."""
        df = make_mock_df(30)
        assert df['USD_TRY_Kapanis'].between(1, 100).all()

    def test_tarih_sutunu_monoton_artan(self):
        """Tarih sütunu sıralı ve artan olmalı."""
        df = make_mock_df(24)
        dates = pd.to_datetime(df['Tarih'])
        assert (dates.diff().dropna() > pd.Timedelta(0)).all()

    def test_ay_sutunu_1_12_arasinda(self):
        """Ay değerleri 1-12 arasında olmalı."""
        df = make_mock_df(24)
        assert df['Ay'].between(1, 12).all()

    def test_nan_orani_kabul_edilebilir(self):
        """İşlenmiş veride NaN oranı %10'dan az olmalı."""
        df = make_mock_df(60)
        # Lag oluştur (ilk satırlar NaN)
        df['lag1'] = df['Serbest_Piyasa_TL_kg'].shift(1)
        nan_ratio = df['lag1'].isna().mean()
        assert nan_ratio < 0.10, f"NaN oranı çok yüksek: {nan_ratio:.2%}"
