import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Dosya Yolları
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

def load_config():
    """config.yaml dosyasını yükler."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_logger(name):
    """Standart loglama formatı döndürür."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(name)

# Konfigürasyon Cache
CFG = load_config()
US_CPI_TABLE = {int(k): v for k, v in CFG.get("us_cpi", {}).items()}
CPI_BAZ_YILI = int(CFG.get("cpi_base_year", 2024))

def reel_usd_to_tl(reel_usd: float, kur: float, yil: int = 2026) -> Tuple[float, float]:
    """Reel USD değerini nominal USD ve TL'ye çevirir."""
    if not US_CPI_TABLE:
        return reel_usd, reel_usd * kur
    
    # CPI Düzeltmesi (Reel -> Nominal)
    cpi_ratio = US_CPI_TABLE.get(CPI_BAZ_YILI, 1.0) / US_CPI_TABLE.get(yil, US_CPI_TABLE.get(CPI_BAZ_YILI, 1.0))
    nominal_usd = reel_usd / cpi_ratio
    return nominal_usd, nominal_usd * kur

# Dizinleri otomatik oluştur
for d in [MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
