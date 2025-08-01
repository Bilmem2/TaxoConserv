@echo off
:: TaxoConserv Auto-Start Service
:: Windows Batch dosyası - sürekli çalışan servis

echo.
echo 🌿 TaxoConserv Auto-Start Service
echo ==================================
echo Web arayüzünü sürekli açık tutar
echo Durdurmak için bu pencereyi kapatın
echo.

:start
echo [%date% %time%] TaxoConserv servisi başlatılıyor...

:: Python servisini başlat
python auto_start_service.py

:: Eğer servis durmuşsa 10 saniye bekle ve yeniden başlat
echo [%date% %time%] Servis durdu. 10 saniye sonra yeniden başlatılacak...
timeout /t 10 /nobreak
echo.
goto start
