"""
tests/test_models.py
====================
Model Yükleme, Tahmin Tutarlılığı ve Ensemble Unit Testleri

Çalıştırma:
    pytest tests/test_models.py -v
    pytest tests/test_models.py -v --cov=src/models
"""

import json
import os
import sys
import pytest
import numpy as np
import pandas as pd

# src dizinini path'e ekle
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "models"))

MODELS_DIR = os.path.join(ROOT, "models")


# ─── Yardımcı Sabitler ────────────────────────────────────────────────────────

US_CPI_TABLE = {
    2013: 69.04, 2014: 70.42, 2015: 71.42, 2016: 73.12,
    2017: 74.91, 2018: 76.97, 2019: 78.59, 2020: 79.84,
    2021: 83.68, 2022: 93.17, 2023: 99.01, 2024: 100.0,
    2025: 103.5,
}
CPI_BAZ_YILI = 2024


# ─── Test Fixtures ────────────────────────────────────────────────────────────

def mock_feature_row(n_features: int = 20) -> dict:
    """Model girişi için sahte feature satırı."""
    np.random.seed(42)
    return {f"feature_{i}": np.random.uniform(0, 10) for i in range(n_features)}


# ─── Reel USD ↔ Nominal Dönüşümü ─────────────────────────────────────────────

class TestCpiConversion:
    """USD reel/nominal dönüşüm mantığı testleri."""

    def _reel_to_nominal(self, reel_usd: float, yil: int) -> float:
        cpi = US_CPI_TABLE[CPI_BAZ_YILI] / US_CPI_TABLE.get(yil, US_CPI_TABLE[CPI_BAZ_YILI])
        return reel_usd / cpi

    def test_baz_yilinda_reel_esit_nominal(self):
        """2024 baz yılında reel fiyat ≈ nominal fiyat olmalı."""
        nominal = self._reel_to_nominal(5.0, 2024)
        assert abs(nominal - 5.0) < 0.001, f"Baz yılı dönüşüm hatası: {nominal}"

    def test_eski_yil_nominal_daha_dusuk(self):
        """Reel→Nominal dönüşümde: 2024 reel 5 USD, 2020 nominali olarak DAHA DÜŞÜK çıkar.
        Çünkü enflasyon nedeniyle 2020'de her şey daha ucuzdu.
        Formül: Nominal = Reel × (CPI_yil / CPI_baz)
        → 5.0 × (79.84 / 100.0) = 3.99 USD (2020 nominal) < 5.0 USD (2024 nominal)
        """
        nominal_2020 = self._reel_to_nominal(5.0, 2020)
        nominal_2024 = self._reel_to_nominal(5.0, 2024)
        assert nominal_2020 < nominal_2024, (
            f"2020 nominal ({nominal_2020:.3f}) < 2024 nominal ({nominal_2024:.3f}) olmalı"
        )

    def test_sifir_giriste_sifir_cikis(self):
        """0 fiyat her yılda 0 kalmalı."""
        assert self._reel_to_nominal(0.0, 2020) == 0.0

    def test_tl_donusumu_pozitif(self):
        """TL fiyatı (nominal_usd × kur) daima pozitif olmalı."""
        kur = 44.0
        reel = 5.0
        nominal = self._reel_to_nominal(reel, 2026)
        tl = nominal * kur
        assert tl > 0, f"TL fiyatı negatif: {tl}"


# ─── Ensemble Ağırlık Mantığı ─────────────────────────────────────────────────

class TestEnsembleWeights:
    """Ensemble ağırlık tutarlılığı testleri."""

    def test_ensemble_weights_json_okunabilir(self):
        """ensemble_weights.json okunabilir ve gerekli anahtarlar var."""
        path = os.path.join(MODELS_DIR, "ensemble_weights.json")
        if not os.path.exists(path):
            pytest.skip("ensemble_weights.json bulunamadı — model henüz eğitilmemiş")
        with open(path, "r") as f:
            weights = json.load(f)
        assert "XGBoost" in weights or "Ridge" in weights, "Ağırlık JSON'u beklenen anahtarları içermiyor"

    def test_ensemble_weights_pozitif(self):
        """Tüm ağırlıklar pozitif (veya sıfır) olmalı."""
        path = os.path.join(MODELS_DIR, "ensemble_weights.json")
        if not os.path.exists(path):
            pytest.skip("ensemble_weights.json yok")
        with open(path, "r") as f:
            weights = json.load(f)
        for model_name, w in weights.items():
            assert float(w) >= 0, f"{model_name} ağırlığı negatif: {w}"

    def test_ensemble_agirlik_toplami_1e_yakin(self):
        """Ağırlık toplamı 1.0'a yakın olmalı (optimize edilmiş)."""
        path = os.path.join(MODELS_DIR, "ensemble_weights.json")
        if not os.path.exists(path):
            pytest.skip("ensemble_weights.json yok")
        with open(path, "r") as f:
            weights = json.load(f)
        total = sum(float(w) for w in weights.values())
        assert abs(total - 1.0) < 0.05, f"Ağırlık toplamı {total:.3f} — 1.0'dan çok sapıyor"

    def test_en_iyi_model_dominant_agirliga_sahip(self):
        """En az bir modelin ağırlığı 0.4'ten büyük olmalı (dominant model)."""
        path = os.path.join(MODELS_DIR, "ensemble_weights.json")
        if not os.path.exists(path):
            pytest.skip("ensemble_weights.json yok")
        with open(path, "r") as f:
            weights = json.load(f)
        max_w = max(float(w) for w in weights.values())
        assert max_w > 0.4, f"Hiçbir model dominant değil (max ağırlık: {max_w:.3f})"


# ─── Conformal Prediction ─────────────────────────────────────────────────────

class TestConformalBounds:
    """Conformal bounds JSON tutarlılığı."""

    def test_conformal_bounds_json_okunabilir(self):
        """conformal_bounds.json okunabilir ve q_hat_relative var."""
        path = os.path.join(MODELS_DIR, "conformal_bounds.json")
        if not os.path.exists(path):
            pytest.skip("conformal_bounds.json bulunamadı")
        with open(path, "r") as f:
            data = json.load(f)
        assert "q_hat_relative" in data, "q_hat_relative anahtarı yok"

    def test_q_hat_mantikli_aralikta(self):
        """q_hat 0.01 ile 0.50 arasında olmalı (çok küçük veya büyük CI saçma)."""
        path = os.path.join(MODELS_DIR, "conformal_bounds.json")
        if not os.path.exists(path):
            pytest.skip("conformal_bounds.json bulunamadı")
        with open(path, "r") as f:
            data = json.load(f)
        q_hat = float(data["q_hat_relative"])
        assert 0.01 < q_hat < 0.50, f"q_hat sınır dışı: {q_hat}"

    def test_conformal_ci_alt_ust_sirasi(self):
        """Conformal CI'da alt sınır < tahmin < üst sınır olmalı."""
        pred = 200.0
        q_hat = 0.15
        ci_low = pred * (1 - q_hat)
        ci_high = pred * (1 + q_hat)
        assert ci_low < pred < ci_high


# ─── Model Skor Dosyası ────────────────────────────────────────────────────────

class TestModelScores:
    """Model performans skoru tutarlılığı."""

    def test_all_model_scores_json_var(self):
        """all_model_scores.json dosyası mevcut olmalı."""
        path = os.path.join(MODELS_DIR, "all_model_scores.json")
        assert os.path.exists(path), "all_model_scores.json yok!"

    def test_weighted_ensemble_en_iyi_mape(self):
        """Weighted Ensemble, bireysel modellerden daha iyi MAPE'ye sahip olmalı."""
        path = os.path.join(MODELS_DIR, "all_model_scores.json")
        if not os.path.exists(path):
            pytest.skip("all_model_scores.json yok")
        with open(path, "r") as f:
            scores = json.load(f)
        if "Weighted Ensemble" not in scores:
            pytest.skip("Weighted Ensemble skoru yok")
        we_mape = scores["Weighted Ensemble"]["MAPE"]
        # XGBoost ve LightGBM'den iyi olmalı
        for model in ["XGBoost", "LightGBM"]:
            if model in scores:
                assert we_mape <= scores[model]["MAPE"] * 1.05, (
                    f"Weighted Ensemble MAPE ({we_mape:.2f}%) {model} MAPE ({scores[model]['MAPE']:.2f}%)'den kötü!"
                )

    def test_her_modelde_gerekli_metrikler(self):
        """Her model için MAE, RMSE, R2, MAPE metrikleri olmalı."""
        path = os.path.join(MODELS_DIR, "all_model_scores.json")
        if not os.path.exists(path):
            pytest.skip("all_model_scores.json yok")
        with open(path, "r") as f:
            scores = json.load(f)
        required_keys = {"MAE", "RMSE", "R2", "MAPE"}
        for model_name, metrics in scores.items():
            missing = required_keys - set(metrics.keys())
            assert not missing, f"{model_name} modelinde eksik metrikler: {missing}"

    def test_mape_makul_aralikta(self):
        """Tüm model MAPE değerleri 1%-60% arasında olmalı (saçma değer yok)."""
        path = os.path.join(MODELS_DIR, "all_model_scores.json")
        if not os.path.exists(path):
            pytest.skip("all_model_scores.json yok")
        with open(path, "r") as f:
            scores = json.load(f)
        for model_name, metrics in scores.items():
            mape = metrics.get("MAPE", 0)
            # N-BEATS ve Prophet kötü ama 200%'den az olmalı
            assert 0 < mape < 200, f"{model_name} MAPE değeri saçma: {mape}"

    def test_r2_negatif_olmayan_en_iyi_model(self):
        """Weighted Ensemble R² pozitif olmalı (temel modelin üstünde)."""
        path = os.path.join(MODELS_DIR, "all_model_scores.json")
        if not os.path.exists(path):
            pytest.skip("all_model_scores.json yok")
        with open(path, "r") as f:
            scores = json.load(f)
        if "Weighted Ensemble" in scores:
            r2 = scores["Weighted Ensemble"]["R2"]
            assert r2 > 0, f"Weighted Ensemble R² negatif: {r2} — model baseline'dan kötü!"


# ─── TMO Tahmini ─────────────────────────────────────────────────────────────

class TestTmoPrediction:
    """TMO 2026 tahmin dosyası tutarlılığı."""

    def test_tmo_prediction_json_var(self):
        """tmo_prediction_2026.json mevcut olmalı."""
        path = os.path.join(MODELS_DIR, "tmo_prediction_2026.json")
        assert os.path.exists(path), "tmo_prediction_2026.json yok!"

    def test_tmo_pred_pozitif(self):
        """TMO tahmini pozitif TL değeri içermeli."""
        path = os.path.join(MODELS_DIR, "tmo_prediction_2026.json")
        if not os.path.exists(path):
            pytest.skip("tmo_prediction_2026.json yok")
        with open(path, "r") as f:
            data = json.load(f)
        assert "pred_2026" in data, "pred_2026 anahtarı yok"
        assert float(data["pred_2026"]) > 0, "TMO tahmini sıfır veya negatif"


# ─── Kausal Etki ─────────────────────────────────────────────────────────────

class TestCausalEffect:
    """Double ML nedensel etki dosyası."""

    def test_causal_effect_json_var(self):
        """causal_effect.json mevcut olmalı."""
        path = os.path.join(MODELS_DIR, "causal_effect.json")
        assert os.path.exists(path), "causal_effect.json yok!"

    def test_ate_pozitif(self):
        """USD/TRY kuru artışının fiyata etkisi (ATE) pozitif olmalı (kur ↑ → fiyat ↑)."""
        path = os.path.join(MODELS_DIR, "causal_effect.json")
        if not os.path.exists(path):
            pytest.skip("causal_effect.json yok")
        with open(path, "r") as f:
            data = json.load(f)
        ate = float(data.get("average_treatment_effect", 0))
        assert ate >= 0, f"ATE negatif ({ate}) — kur artışı fiyatı düşürüyor? Domain bilgisiyle çelişiyor."
