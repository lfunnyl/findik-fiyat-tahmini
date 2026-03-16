@echo off
setlocal

:: Fındık Fiyatı Tahmin Projesi - Otomatik Veri Güncelleme
:: Bu dosya Windows Task Scheduler (Görev Zamanlayıcı) ile 
:: her ayın belirli bir gününde (örn: her ayın 1'i ve 15'i) çalıştırılmak üzere tasarlanmıştır.

echo ========================================================
echo FINDIK FIYAT TAHMINI: OTOMATIK GUNCELLEME BASLATILIYOR...
echo Tarih: %date% %time%
echo ========================================================

:: Scriptin bulundugu klasoru base dizin kabul et
cd /D "%~dp0"

:: Python ile pipeline scriptini calistir
python src\data\update_pipeline.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo PIPELINE BASARIYLA TAMAMLANDI! (Loglar pipeline.log dosyasinda)
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo HATA: PIPELINE BEKLENMEYEN BIR HATA ILE SONLANDI.
    echo Lutfen pipeline.log dosyasini kontrol edin.
    echo ========================================================
)

endlocal
pause
