@echo off
echo ================================================================================
echo                     🌿 TaxoConserv Web Interface Launcher
echo                  Taxonomic Conservation Analysis Tool v1.0.0
echo ================================================================================
echo.
echo 🚀 Starting TaxoConserv Web Interface...
echo 📍 Location: %~dp0
echo 🌐 Opening browser at: http://localhost:8501
echo.
echo ⚡ Press Ctrl+C to stop the server
echo 💡 Close this window to stop the application
echo.
echo ================================================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo 💡 Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Streamlit is not installed
    echo 🔧 Installing Streamlit...
    pip install streamlit plotly scipy
    if %errorlevel% neq 0 (
        echo ❌ Failed to install Streamlit
        pause
        exit /b 1
    )
)

REM Start the web interface
echo ✅ Starting Streamlit server...
echo.
streamlit run web_taxoconserv.py --server.port 8501 --server.address localhost

echo.
echo 🛑 Server stopped
pause
