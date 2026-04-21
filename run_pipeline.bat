@echo off
echo =======================================================
echo FINDIK FIYATI TAHMIN SISTEMI - TAM BORU HATTI (PIPELINE)
echo =======================================================
echo.

echo [1/4] Ortam bagimliliklari kontrol ediliyor...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [HATA] Bagimliliklar yuklenemedi.
    exit /b %errorlevel%
)

echo [2/4] Modeller Egitiliyor (XGBoost, LightGBM, CatBoost)...
python src\models\train_model.py
if %errorlevel% neq 0 (
    echo [HATA] Model egitimi sirasinda hata olustu.
    exit /b %errorlevel%
)

echo [3/4] Hata (Residual) Analizi calistiriliyor...
python src\evaluation\residual_analysis.py
if %errorlevel% neq 0 (
    echo [UYARI] Hata analizi sirasinda bazi sorunlar yasandi ama devam ediliyor.
)

echo [4/4] Causal Inference (Double ML) Analizi Guncelleniyor...
python src\evaluation\causal_inference.py
if %errorlevel% neq 0 (
    echo [UYARI] Causal Inference surecinde hata olustu.
)

echo.
echo =======================================================
echo TUM ADIMLAR TAMAMLANDI!
echo Streamlit arayuzunu baslatmak icin:
echo streamlit run app.py
echo =======================================================
pause
