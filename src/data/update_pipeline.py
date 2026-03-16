"""
update_pipeline.py
==================
Otomatik Veri Güncelleme ve Model Eğitimi Pipeline'ı

Bu script, projedeki tüm veri çekme (scraper) scriptlerini sırayla çalıştırır, 
ardından features/build_features.py ile verileri birleştirir ve son olarak 
tüm makine öğrenmesi modellerini yeniden eğitir.

Kullanım:
  python src/data/update_pipeline.py
"""

import os
import sys
import subprocess
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pipeline.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Çalıştırılacak Scriptlerin Sırası
PIPELINE_STEPS = [
    {
        "name": "1. Veri Toplama (Scrapers)",
        "scripts": [
            "src/data/findik_fiyat_scraper.py",
            "src/data/fao_findik_scraper.py",
            "src/data/makro_veriler_scraper.py",
            "src/data/ekstra_makro_scraper.py",
            "src/data/tufe_scraper.py",
            "src/data/ihracat_findik.py",
            "src/data/karadeniz_iklim_scraper.py",
            "src/data/rekolte_arz_talep.py",
            "src/data/asgeriücret.py"
        ],
        "fatal_on_fail": False  # Bir scraper hata verirse diğerleri çalışmaya devam etsin
    },
    {
        "name": "2. Veri İşleme (Feature Engineering)",
        "scripts": [
            "src/features/build_features.py"
        ],
        "fatal_on_fail": True   # Master tablo oluşmazsa modeller eğitilemez
    },
    {
        "name": "3. OOS Gerçek Test Skoru Takibi",
        "scripts": [
            "src/evaluation/track_performance.py"
        ],
        "fatal_on_fail": False  # Track logu kopsa da model eğitimi devam etsin
    },
    {
        "name": "4. Model Eğitimi (Retrain)",
        "scripts": [
            "src/models/train_model.py",
            "src/models/advanced_models.py",
            "src/models/tmo_model.py"
        ],
        "fatal_on_fail": True
    }
]


def run_script(script_path):
    """Verilen Python scriptini subprocess olarak çalıştırır."""
    full_path = os.path.join(BASE_DIR, script_path)
    if not os.path.exists(full_path):
        logger.error(f"Dosya bulunamadi: {full_path}")
        return False
        
    logger.info(f"==> Calistiriliyor: {script_path}")
    
    # sys.executable kullanarak ayni Python environment'inda calismasini sagliyoruz
    result = subprocess.run(
        [sys.executable, full_path],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if result.returncode == 0:
        logger.info(f"[BASARILI] {script_path}")
        return True
    else:
        logger.error(f"[HATA] {script_path} exit_code={result.returncode}")
        logger.error("Hata Detayi:\n" + result.stdout)
        return False


def main():
    start_time = time.time()
    logger.info("==================================================")
    logger.info("FINDIK FIYAT TAHMINI - UPDATE PIPELINE BASLADI")
    logger.info("==================================================")
    
    total_scripts = 0
    failed_scripts = 0
    
    for phase in PIPELINE_STEPS:
        logger.info(f"\n--- {phase['name']} ---")
        for script in phase["scripts"]:
            total_scripts += 1
            success = run_script(script)
            if not success:
                failed_scripts += 1
                if phase.get("fatal_on_fail", False):
                    logger.critical(f"FATAL HATA: Kritik asama basarisiz oldu. Pipeline durduruluyor ({script})")
                    sys.exit(1)
            time.sleep(1)  # Scriptler arası ufak gecikme (API limitleri için)
            
    elapsed = time.time() - start_time
    logger.info("\n==================================================")
    logger.info(f"PIPELINE TAMAMLANDI - Gecen Sure: {elapsed:.1f} saniye")
    logger.info(f"Toplam Calistirma: {total_scripts} | Basarisiz: {failed_scripts}")
    logger.info("==================================================")
    
    if failed_scripts > 0:
        sys.exit(1)
        
if __name__ == "__main__":
    main()
