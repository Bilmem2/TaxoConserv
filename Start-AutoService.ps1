# TaxoConserv Auto-Start Service
# PowerShell script for continuous service

Write-Host ""
Write-Host "🌿 TaxoConserv Auto-Start Service" -ForegroundColor Green
Write-Host "=================================="
Write-Host "Web arayüzünü sürekli açık tutar"
Write-Host "Durdurmak için Ctrl+C kullanın"
Write-Host ""

# Infinite loop with restart capability
while ($true) {
    try {
        Write-Host "[$(Get-Date)] TaxoConserv servisi başlatılıyor..." -ForegroundColor Yellow
        
        # Start the Python service
        $process = Start-Process -FilePath "python" -ArgumentList "auto_start_service.py" -Wait -PassThru
        
        Write-Host "[$(Get-Date)] Servis durdu (Exit Code: $($process.ExitCode)). 10 saniye sonra yeniden başlatılacak..." -ForegroundColor Red
        Start-Sleep -Seconds 10
        
    }
    catch {
        Write-Host "[$(Get-Date)] Hata: $_" -ForegroundColor Red
        Write-Host "10 saniye sonra yeniden denenecek..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
    }
}
