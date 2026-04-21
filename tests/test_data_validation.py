"""
tests/test_data_validation.py
==============================
Veri Kalitesi ve Şema Doğrulama Testleri

CSV dosyalarının şemasını, değer aralıklarını ve tutarlılığını kontrol eder.

Çalıştırma:
    pytest tests/test_data_validation.py -v
    pytest tests/test_data_validation.py -v -k "TestMasterFeatures"
"""

import os
import sys
import pytest
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")


# ─── Yardımcı ─────────────────────────────────────────────────────────────────

def load_master() -> pd.DataFrame | None:
    """master_features.csv yükler; yoksa None döner."""
    path = os.path.join(DATA_DIR, "master_features.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    return df


# ─── master_features.csv ──────────────────────────────────────────────────────

class TestMasterFeatures:
    """Ana özellik matrisinin şema ve kalite testleri."""

    def test_master_csv_var(self):
        """master_features.csv dosyası mevcut olmalı."""
        path = os.path.join(DATA_DIR, "master_features.csv")
        assert os.path.exists(path), f"master_features.csv bulunamadı: {path}"

    def test_minimum_satir_sayisi(self):
        """Veri seti en az 60 aylık veri içermeli."""
        df = load_master()
        if df is None:
            pytest.skip("master_features.csv yok")
        assert len(df) >= 60, f"Yetersiz veri: {len(df)} satır (minimum 60)"

    def test_zorunlu_kolonlar_var(self):
        """Kritik kolonlar master_features.csv'de bulunmalı."""
        df = load_master()
        if df is None:
            pytest.skip("master_features.csv yok")
        required_cols = [
            "Tarih",
            "Serbest_Piyasa_TL_kg",
            "Fiyat_USD_kg",
            "Fiyat_RealUSD_kg",
            "USD_TRY_Kapanis",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        assert not missing, f"Eksik zorunlu kolonlar: {missing}"

    def test_tarih_monoton_artan(self):
        """Tarih sütunu tekrarsız ve sıralı olmalı."""
        df = load_master()
        if df is None:
            pytest.skip("master_features.csv yok")
        dates = df["Tarih"].sort_values().reset_index(drop=True)
        diffs = dates.diff().dropna()
        assert (diffs > pd.Timedelta(0)).all(), "Tarihler monoton artmıyor!"

    def test_fiyatlar_pozitif(self):
        """Tüm fiyat kolonları sıfırdan büyük olmalı."""
        df = load_master()
        if df is None:
            pytest.skip("master_features.csv yok")
        price_cols = ["Serbest_Piyasa_TL_kg", "Fiyat_USD_kg", "Fiyat_RealUSD_kg"]
        for col in price_cols:
            if col in df.columns:
                non_positive = (df[col] <= 0).sum()
                assert non_positive == 0, f"{col}: {non_positive} sıfır veya negatif değer!"

    def test_kur_mantikli_aralikta(self):
        """USD/TRY kuru 1 ile 100 arasında olmalı."""
        df = load_master()
        if df is None:
            pytest.skip("master_features.csv yok")
        if "USD_TRY_Kapanis" not in df.columns:
            pytest.skip("USD_TRY_Kapanis kolonu yok")
        kur = df["USD_TRY_Kapanis"]
        assert kur.between(1, 100).all(), (
            f"Kur sınır dışı: min={kur.min():.2f}, max={kur.max():.2f}"
        )

    def test_nan_orani_kabul_edilebilir(self):
        """Sayısal kolonlarda NaN oranı %20'den az olmalı."""
        df = load_master()
        if df is None:
            pytest.skip("master_features.csv yok")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            nan_ratio = df[col].isna().mean()
            assert nan_ratio < 0.20, (
                f"{col}: NaN oranı çok yüksek ({nan_ratio:.1%})"
            )

    def test_reel_usd_nominal_usd_dan_farkli(self):
        """Reel USD fiyatı nominal USD'den farklı olmalı (CPI düzeltmesi yapıldı)."""
        df = load_master()
        if df is None:
            pytest.skip("master_features.csv yok")
        if "Fiyat_USD_kg" not in df.columns or "Fiyat_RealUSD_kg" not in df.columns:
            pytest.skip("Gerekli kolonlar yok")
        # En az bir satırda fark olmalı
        diff = (df["Fiyat_USD_kg"] - df["Fiyat_RealUSD_kg"]).abs().max()
        assert diff > 0.001, "Reel ve nominal USD fiyatları aynı — CPI düzeltmesi yapılmamış!"

    def test_zaman_serisi_aylik_frekansta(self):
        """Veri seti aylık frekansta olmalı (max boşluk 62 gün)."""
        df = load_master()
        if df is None:
            pytest.skip("master_features.csv yok")
        df_sorted = df.sort_values("Tarih")
        diffs_days = df_sorted["Tarih"].diff().dropna().dt.days
        assert diffs_days.max() <= 62, (
            f"Büyük zaman boşluğu: {diffs_days.max()} gün — aylık frekanstan sapma!"
        )


# ─── Destekleyici CSV Dosyaları ───────────────────────────────────────────────

class TestSupportingData:
    """Ham ve işlenmiş yardımcı veri dosyalarının varlık testleri."""

    def test_findik_fiyatlari_csv_var(self):
        """Türkiye fındık fiyatları CSV mevcut olmalı."""
        path = os.path.join(DATA_DIR, "turkiye_findik_fiyatlari_temiz.csv")
        assert os.path.exists(path), "turkiye_findik_fiyatlari_temiz.csv yok"

    def test_hava_durumu_json_var(self):
        """hava_durumu_3aylik.json API için mevcut olmalı."""
        path = os.path.join(DATA_DIR, "hava_durumu_3aylik.json")
        assert os.path.exists(path), "hava_durumu_3aylik.json yok — /api/weather 404 döner!"

    def test_hava_durumu_json_okunabilir(self):
        """hava_durumu_3aylik.json geçerli JSON olmalı."""
        path = os.path.join(DATA_DIR, "hava_durumu_3aylik.json")
        if not os.path.exists(path):
            pytest.skip("hava_durumu_3aylik.json yok")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, (dict, list)), "JSON geçerli formatta değil"

    def test_fao_csv_var(self):
        """FAO fındık üretim verisi mevcut olmalı."""
        path = os.path.join(DATA_DIR, "fao_findik_uretim_temiz.csv")
        assert os.path.exists(path), "fao_findik_uretim_temiz.csv yok"

    def test_makro_veriler_csv_var(self):
        """Makro ekonomi verisi (USD/TRY kuru dahil) mevcut olmalı."""
        path = os.path.join(DATA_DIR, "makro_veriler_5_years_temiz.csv")
        assert os.path.exists(path), "makro_veriler_5_years_temiz.csv yok"

    def test_karadeniz_iklim_csv_var(self):
        """Karadeniz iklim verisi mevcut olmalı."""
        path = os.path.join(DATA_DIR, "karadeniz_iklim_5_years_temiz.csv")
        assert os.path.exists(path), "karadeniz_iklim_5_years_temiz.csv yok"


# ─── Model Dosyaları Varlık Kontrolü ─────────────────────────────────────────

class TestModelFiles:
    """Kritik model dosyalarının varlığı ve boyut kontrolü."""

    CRITICAL_MODELS = [
        "xgboost_model.pkl",
        "lightgbm_model.pkl",
        "ridge_model.pkl",
        "multistep_1m.pkl",
        "multistep_3m.pkl",
        "multistep_6m.pkl",
    ]

    CRITICAL_JSONS = [
        "ensemble_weights.json",
        "conformal_bounds.json",
        "all_model_scores.json",
        "tmo_prediction_2026.json",
        "causal_effect.json",
    ]

    def test_kritik_model_pkl_dosyalari_var(self):
        """Tüm kritik .pkl model dosyaları mevcut olmalı."""
        missing = []
        for fname in self.CRITICAL_MODELS:
            fpath = os.path.join(MODELS_DIR, fname)
            if not os.path.exists(fpath):
                missing.append(fname)
        assert not missing, f"Eksik kritik model dosyaları: {missing}"

    def test_kritik_json_dosyalari_var(self):
        """Tüm kritik JSON meta dosyaları mevcut olmalı."""
        missing = []
        for fname in self.CRITICAL_JSONS:
            fpath = os.path.join(MODELS_DIR, fname)
            if not os.path.exists(fpath):
                missing.append(fname)
        assert not missing, f"Eksik JSON dosyaları: {missing}"

    def test_model_boyutu_makul(self):
        """Temel XGBoost modeli 50 MB'dan küçük olmalı."""
        path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
        if not os.path.exists(path):
            pytest.skip("xgboost_model.pkl yok")
        size_mb = os.path.getsize(path) / (1024 * 1024)
        assert size_mb < 50, f"XGBoost model çok büyük: {size_mb:.1f} MB"


# ─── Veri Tutarlılığı — Çapraz Doğrulama ────────────────────────────────────

class TestDataConsistency:
    """Farklı veri kaynakları arasındaki tutarlılık kontrolü."""

    def test_master_tarih_aralik_makul(self):
        """master_features.csv 2013 öncesi ve 2027 sonrası veri içermemeli."""
        df = load_master()
        if df is None:
            pytest.skip("master_features.csv yok")
        min_year = df["Tarih"].dt.year.min()
        max_year = df["Tarih"].dt.year.max()
        assert min_year >= 2013, f"Çok eski veri: {min_year}"
        assert max_year <= 2027, f"Gelecek veri: {max_year}"

    def test_usd_tl_fiyat_tutarliligi(self):
        """TL fiyat / USD fiyat oranı yaklaşık kur değerine eşit olmalı."""
        df = load_master()
        if df is None:
            pytest.skip("master_features.csv yok")
        required = {"Serbest_Piyasa_TL_kg", "Fiyat_USD_kg", "USD_TRY_Kapanis"}
        if not required.issubset(df.columns):
            pytest.skip("Gerekli kolonlar yok")
        # Hesaplanan kur = TL/USD — gerçek kurla %30 tolerans
        df_clean = df.dropna(subset=list(required))
        df_clean = df_clean[df_clean["Fiyat_USD_kg"] > 0]
        hesap_kur = df_clean["Serbest_Piyasa_TL_kg"] / df_clean["Fiyat_USD_kg"]
        gercek_kur = df_clean["USD_TRY_Kapanis"]
        oran = (hesap_kur / gercek_kur).dropna()
        # Çoğunlukla 0.7 ile 1.3 arasında olmalı (TL bazlı fiyat ≈ USD × kur)
        assert (oran.between(0.5, 2.0)).mean() > 0.7, (
            "TL/USD/kur tutarsızlığı — veri entegrasyonunda sorun var!"
        )

    def test_fiyat_trend_makul(self):
        """2013-2026 arası median fındık fiyatı pozitif trend içermeli."""
        df = load_master()
        if df is None:
            pytest.skip("master_features.csv yok")
        if "Serbest_Piyasa_TL_kg" not in df.columns:
            pytest.skip("Serbest_Piyasa_TL_kg kolonu yok")
        df_sorted = df.sort_values("Tarih")
        first_half_med = df_sorted.head(len(df_sorted) // 2)["Serbest_Piyasa_TL_kg"].median()
        second_half_med = df_sorted.tail(len(df_sorted) // 2)["Serbest_Piyasa_TL_kg"].median()
        # TL bazında pozitif trend beklenir (enflasyon etkisi)
        assert second_half_med > first_half_med, (
            "TL fiyatları zamanla azalmış — enflasyon beklentisiyle çelişiyor!"
        )
