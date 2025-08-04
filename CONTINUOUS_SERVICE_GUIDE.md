# TaxoConserv Sürekli Açık Kalma Rehberi

## 🔄 Sürekli Çalışan Web Arayüzü Seçenekleri

TaxoConserv web arayüzünün sürekli açık kalması için 4 farklı yöntem sunulmaktadır:

### 1. 🚀 Otomatik Servis (Önerilen)

**Kullanım:**
```bash
# Otomatik servisi başlat
python auto_start_service.py
```

**Özellikler:**
- ✅ Otomatik yeniden başlatma
- ✅ Log kaydı
- ✅ Hata durumunda kendini onarma
- ✅ Maximum restart limiti (güvenlik)

### 2. 🖥️ Windows Batch Dosyası

**Kullanım:**
```bash
# Batch dosyasını çift tıklayarak çalıştır
start_auto_service.bat
```

**Özellikler:**
- ✅ Kolay kullanım (çift tık)
- ✅ Otomatik yeniden başlatma
- ✅ Windows konsol çıktısı

### 3. ⚡ PowerShell Script

**Kullanım:**
```powershell
# PowerShell'i yönetici olarak açın
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\Start-AutoService.ps1
```

**Özellikler:**
- ✅ Gelişmiş Windows entegrasyonu
- ✅ Renkli konsol çıktısı
- ✅ PowerShell error handling

### 4. 🛠️ Windows Task Scheduler

**Manuel Kurulum:**
1. **Task Scheduler** açın (`taskschd.msc`)
2. **Create Basic Task** seçin
3. **Name:** "TaxoConserv AutoStart"
4. **Trigger:** "When the computer starts"
5. **Action:** "Start a program"
6. **Program:** `python`
7. **Arguments:** `auto_start_service.py`
8. **Start in:** `C:\Users\can_t\Downloads\TaxoConserv`

## 🔧 Kurulum ve Kullanım

### Hızlı Başlatma
```bash
# 1. Otomatik servisi başlat (önerilen)
python auto_start_service.py

# VEYA Windows kullanıcıları için
start_auto_service.bat
```

### Erişim Adresleri
- **Desktop:** http://localhost:8080
- **Mobil/Diğer cihazlar:** http://192.168.0.13:8080

### Servisi Durdurma
- **Konsol:** `Ctrl + C`
- **Batch:** Pencereyi kapat
- **PowerShell:** `Ctrl + C`
- **Task Scheduler:** Task'ı disable et

## 📊 Servis Özellikleri

| Özellik | Otomatik Servis | Batch | PowerShell | Task Scheduler |
|---------|-----------------|-------|------------|----------------|
| Otomatik başlatma | ✅ | ✅ | ✅ | ✅ |
| Sistem başlangıcında | ❌ | ❌ | ❌ | ✅ |
| Log kaydı | ✅ | ❌ | ❌ | ✅ |
| Hata toleransı | ✅ | ✅ | ✅ | ✅ |
| Kolay durdurma | ✅ | ✅ | ✅ | ❌ |

## 🚨 Önemli Notlar

1. **Güvenlik:** Firewall kurallarının aktif olduğundan emin olun
2. **Performans:** Büyük veri setleri için sistem kaynaklarını izleyin
3. **Ağ:** Router ayarlarınızda port 8080'in açık olduğunu kontrol edin
4. **Güncellemeler:** Servis çalışırken kod değişiklikleri otomatik yüklenmez

## 🔍 Sorun Giderme

### Servis Başlamıyor
```bash
# Port kullanımını kontrol et
netstat -an | findstr :8080

# Streamlit process'leri temizle
taskkill /f /im streamlit.exe
```

### Mobil Erişim Yok
```bash
# IP adresini doğrula
ipconfig | findstr "IPv4"

# Firewall kuralını kontrol et
netsh advfirewall firewall show rule name="TaxoConserv"
```

### Log Dosyası Kontrolü
```bash
# Servis log'larını görüntüle
type taxoconserv_service.log
```

## 📱 Önerilen Konfigürasyon

**Akademik/Araştırma Kullanımı:**
- Otomatik servis + Task Scheduler kombinasyonu
- Günlük log backup'ı
- Düzenli sistem restart'ı (haftalık)

**Demo/Sunum:**
- Batch dosyası (hızlı başlatma)
- Manuel durdurma kontrolü

**Production/Sürekli Kullanım:**
- Task Scheduler + otomatik servis
- System monitoring
- Firewall optimization
