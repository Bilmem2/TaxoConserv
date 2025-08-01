# TaxoConserv Web Interface Launcher (PowerShell)
# Modern launcher script for Windows PowerShell

param(
    [switch]$NoAutoOpen = $false,
    [int]$Port = 8501
)

# Set colors
$Host.UI.RawUI.WindowTitle = "🌿 TaxoConserv Web Interface"

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "                    🌿 TaxoConserv Web Interface Launcher" -ForegroundColor Cyan
Write-Host "                 Taxonomic Conservation Analysis Tool v1.0.0" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""

# Function to test if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Function to test if a Python package is installed
function Test-PythonPackage($packagename) {
    try {
        $result = python -c "import $packagename; print('OK')" 2>$null
        return $result -eq "OK"
    }
    catch {
        return $false
    }
}

# Check Python installation
Write-Host "🔍 Checking Python installation..." -ForegroundColor Yellow
if (-not (Test-Command "python")) {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "💡 Please install Python from https://python.org" -ForegroundColor Blue
    Read-Host "Press Enter to exit"
    exit 1
}

$pythonVersion = python --version
Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green

# Check dependencies
Write-Host "🔍 Checking dependencies..." -ForegroundColor Yellow
$requiredPackages = @('streamlit', 'plotly', 'scipy', 'pandas', 'numpy', 'matplotlib', 'seaborn')
$missingPackages = @()

foreach ($package in $requiredPackages) {
    if (Test-PythonPackage $package) {
        Write-Host "✅ $package - OK" -ForegroundColor Green
    } else {
        Write-Host "❌ $package - Missing" -ForegroundColor Red
        $missingPackages += $package
    }
}

# Install missing packages
if ($missingPackages.Count -gt 0) {
    Write-Host "🔧 Installing missing packages: $($missingPackages -join ', ')" -ForegroundColor Yellow
    try {
        python -m pip install $missingPackages
        Write-Host "✅ All dependencies installed successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        Write-Host "💡 Please run manually: pip install $($missingPackages -join ' ')" -ForegroundColor Blue
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "✅ All dependencies are installed!" -ForegroundColor Green
}

Write-Host ""

# Check if web script exists
$webScript = "web_taxoconserv.py"
if (-not (Test-Path $webScript)) {
    Write-Host "❌ ERROR: $webScript not found!" -ForegroundColor Red
    Write-Host "Expected location: $(Get-Location)\$webScript" -ForegroundColor Blue
    Read-Host "Press Enter to exit"
    exit 1
}

# Start the web interface
Write-Host "🚀 Starting TaxoConserv Web Interface..." -ForegroundColor Green
Write-Host "📍 Location: $(Get-Location)" -ForegroundColor Blue
Write-Host "🌐 Web interface will be available at: http://localhost:$Port" -ForegroundColor Blue
Write-Host ""
Write-Host "⚡ Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "💡 Close this window to stop the application" -ForegroundColor Yellow
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""

# Open browser after delay (unless disabled)
if (-not $NoAutoOpen) {
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 3
        Start-Process "http://localhost:$using:Port"
    } | Out-Null
}

# Start Streamlit
try {
    streamlit run $webScript --server.port $Port --server.address localhost
}
catch {
    Write-Host ""
    Write-Host "❌ Error starting server: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🛑 Server stopped" -ForegroundColor Yellow
Read-Host "Press Enter to exit"
