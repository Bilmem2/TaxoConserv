"""
TaxoConserv Web Interface - Auto Start Service
Sürekli çalışan web arayüzü servisi
"""

import subprocess
import time
import os
import sys
import logging
from datetime import datetime

# Logging konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('taxoconserv_service.log'),
        logging.StreamHandler()
    ]
)

class TaxoConservService:
    def __init__(self, port=8080, host="0.0.0.0"):
        self.port = port
        self.host = host
        self.process = None
        self.restart_count = 0
        self.max_restarts = 10
        
    def start_server(self):
        """Streamlit sunucusunu başlat"""
        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                "web_taxoconserv.py",
                "--server.address", self.host,
                "--server.port", str(self.port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
            
            logging.info(f"Starting TaxoConserv server on {self.host}:{self.port}")
            self.process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logging.info(f"Server started with PID: {self.process.pid}")
            return True
            
        except Exception as e:
            logging.error(f"Error starting server: {e}")
            return False
    
    def is_server_running(self):
        """Sunucunun çalışıp çalışmadığını kontrol et"""
        if self.process is None:
            return False
        
        poll_result = self.process.poll()
        return poll_result is None
    
    def stop_server(self):
        """Sunucuyu durdur"""
        if self.process:
            logging.info("Stopping server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
    
    def restart_server(self):
        """Sunucuyu yeniden başlat"""
        logging.info("Restarting server...")
        self.stop_server()
        time.sleep(5)  # 5 saniye bekle
        return self.start_server()
    
    def monitor_and_restart(self):
        """Sunucuyu sürekli izle ve gerekirse yeniden başlat"""
        logging.info("Starting TaxoConserv monitoring service...")
        
        while self.restart_count < self.max_restarts:
            if not self.is_server_running():
                logging.warning("Server is not running. Attempting restart...")
                
                if self.restart_server():
                    self.restart_count += 1
                    logging.info(f"Server restarted successfully (attempt {self.restart_count})")
                else:
                    logging.error("Failed to restart server")
                    self.restart_count += 1
            
            # Her 30 saniyede bir kontrol et
            time.sleep(30)
        
        logging.error(f"Maximum restart attempts ({self.max_restarts}) reached. Stopping service.")

def main():
    """Ana fonksiyon"""
    print("🌿 TaxoConserv Auto-Start Service")
    print("==================================")
    print("Bu servis web arayüzünü sürekli açık tutar.")
    print("Durdurmak için Ctrl+C kullanın.\n")
    
    service = TaxoConservService()
    
    try:
        # İlk başlatma
        if service.start_server():
            print(f"✅ Server başlatıldı: http://192.168.0.13:8080")
            print("📱 Mobil erişim: http://192.168.0.13:8080")
            print("🔄 Otomatik yeniden başlatma aktif\n")
            
            # İzleme başlat
            service.monitor_and_restart()
        else:
            print("❌ Server başlatılamadı!")
            
    except KeyboardInterrupt:
        print("\n🛑 Servis durduruluyor...")
        service.stop_server()
        print("✅ Servis durduruldu.")
    except Exception as e:
        logging.error(f"Service error: {e}")
        service.stop_server()

if __name__ == "__main__":
    main()
