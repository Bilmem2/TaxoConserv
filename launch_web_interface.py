#!/usr/bin/env python3
"""
TaxoConserv Web Interface Launcher
Easy launcher script for the Streamlit web application
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    """Print startup banner."""
    print("=" * 80)
    print("                    🌿 TaxoConserv Web Interface Launcher")
    print("                 Taxonomic Conservation Analysis Tool v2.0.0")
    print("=" * 80)
    print()

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = ['streamlit', 'plotly', 'scipy', 'pandas', 'numpy', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("💡 Please run: pip install streamlit plotly scipy pandas numpy matplotlib seaborn")
            input("Press Enter to exit...")
            sys.exit(1)
    else:
        print("✅ All dependencies are installed!")
    
    print()

def start_web_interface():
    """Start the Streamlit web interface."""
    script_dir = Path(__file__).parent
    web_script = script_dir / "web_taxoconserv.py"
    
    if not web_script.exists():
        print("❌ ERROR: web_taxoconserv.py not found!")
        print(f"Expected location: {web_script}")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("🚀 Starting TaxoConserv Web Interface...")
    print("📍 Location:", script_dir)
    print("🌐 Opening browser at: http://localhost:8501")
    print()
    print("⚡ Press Ctrl+C to stop the server")
    print("💡 Close this window to stop the application")
    print()
    print("=" * 80)
    print()
    
    # Change to script directory
    os.chdir(script_dir)
    
    # Start browser after a short delay
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'web_taxoconserv.py',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
    
    print("\n🛑 Server stopped")
    input("Press Enter to exit...")

def main():
    """Main launcher function."""
    print_banner()
    check_dependencies()
    start_web_interface()

if __name__ == "__main__":
    main()
