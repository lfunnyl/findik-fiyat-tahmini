@echo off
set PYTHONPATH=.
echo =======================================================
echo FINDIK FIYATI TAHMIN SISTEMI - TAM BORU HATTI (PIPELINE)
echo =======================================================
echo.

echo [1/3] Ortam bagimliliklari kontrol ediliyor...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [HATA] Bagimliliklar yuklenemedi.
    exit /b %errorlevel%
)

echo [2/4] Ham Veriler isleniyor ve Ozellikler Turetiliyor...
python src\features\build_features.py
if %errorlevel% neq 0 (
    echo [HATA] Veri isleme sirasinda hata olustu.
    exit /b %errorlevel%
)

echo [3/4] Modeller Egitiliyor (XGBoost, LightGBM, CatBoost)...
python src\models\train_model.py
if %errorlevel% neq 0 (
    echo [HATA] Model egitimi sirasinda hata olustu.
    exit /b %errorlevel%
)

echo [4/4] SHAP Analizleri Uretiliyor...
python scripts\generate_shap.py
if %errorlevel% neq 0 (
    echo [UYARI] SHAP guncellemesinde sorun yasandi.
)

echo.
echo =======================================================
echo TUM ADIMLAR TAMAMLANDI! (Modeller egitildi ve kaydedildi)
echo.
echo API'yi baslatmak icin:
echo python api\main.py
echo.
echo Dashboard'u (Arayuz) baslatmak icin:
echo cd dashboard ^&^& npm run dev
echo =======================================================
pause
