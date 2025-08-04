# TaxoConserv Cloud Deployment Guide
# 24/7 Erişim için Bulut Hosting Seçenekleri

## 🌐 Streamlit Cloud (Ücretsiz - Önerilen)

### Kurulum Adımları:

**Seçenek 1: Mevcut Repository Kullan (Önerilen)**
1. **Mevcut GitHub Repository'nizi Güncelleyin**
   - Cloud deployment dosyalarını mevcut repo'ya ekleyin
   - `requirements-cloud.txt` ve `Procfile` dosyaları eklendi
   - GitHub'a push yapın

2. **Streamlit Cloud'a Deploy Et**
   - https://share.streamlit.io adresine git
   - GitHub ile giriş yap
   - "New app" → **Mevcut repository'nizi** seç
   - Main file: `web_taxoconserv.py`

**Seçenek 2: Yeni Repository (Sadece Web İçin)**
1. **Ayrı Web Repository Oluştur**
   - GitHub'da yeni repo: `TaxoConserv-Web`
   - Sadece web interface dosyalarını kopyala

3. **Otomatik Deployment**
   - GitHub'a push → Otomatik güncelleme
   - 24/7 erişim sağlanır

### Avantajlar:
- ✅ Tamamen ücretsiz
- ✅ 24/7 erişim
- ✅ Otomatik güncelleme
- ✅ HTTPS güvenlik
- ✅ Global erişim

## ☁️ Heroku (Ücretsiz Tier)

### requirements.txt için:
```txt
streamlit>=1.28.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
plotly>=5.15.0
numpy>=1.23.0
```

### Procfile:
```
web: streamlit run web_taxoconserv.py --server.port=$PORT --server.address=0.0.0.0
```

## 🐳 Docker + Free Hosting

### Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "web_taxoconserv.py", "--server.address=0.0.0.0"]
```

## 💻 Raspberry Pi (Ev Sunucusu)

### Avantajlar:
- 24/7 çalışır
- Düşük elektrik tüketimi
- Tam kontrol
- Yerel ağ erişimi

### Kurulum:
```bash
# Raspberry Pi'da
sudo apt update
sudo apt install python3-pip
pip3 install streamlit pandas matplotlib seaborn scipy
```

## 🏠 Mini PC / NUC (Ev Sunucusu)

### Önerilen Konfigürasyon:
- Intel NUC veya benzeri mini PC
- Ubuntu Server
- Auto-start konfigürasyonu
- Port forwarding (router)

## 📱 VPS (Virtual Private Server)

### Ücretsiz/Düşük Maliyetli Seçenekler:
1. **Oracle Cloud Always Free Tier**
2. **Google Cloud $300 kredi**
3. **AWS EC2 Free Tier**
4. **DigitalOcean $100 kredi**

## 🚀 Streamlit Cloud Deployment Süreci

### Adımlar:
1. **https://share.streamlit.io** → "Sign in with GitHub"
2. **"New app"** → Repository: `Bilmem2/TaxoConserv`
3. **Branch:** `master` **Main file:** `web_taxoconserv.py`
4. **"Deploy!"**

### Beklenen Süre: 3-5 dakika

## 🔧 Muhtemel Deployment Sorunları

### Dependency Hatası
```bash
ModuleNotFoundError: No module named 'xyz'
```
**Çözüm:** requirements-cloud.txt'ye eksik modülü ekleyin

### Port Hatası
```bash
Port 8501 is already in use
```
**Çözüm:** Streamlit Cloud otomatik halleder

### File Not Found
```bash
FileNotFoundError: web_taxoconserv.py
```
**Çözüm:** Main file path'i kontrol edin

## ✅ Başarılı Deployment Sonrası

**🎉 TaxoConserv Cloud URL:** https://taxoconserv.streamlit.app/

### Deployment Özellikleri:
- ✅ 24/7 erişim aktif
- ✅ Otomatik güncelleme (GitHub push → canlı yayın)
- ✅ HTTPS güvenli bağlantı
- ✅ Global erişim
- ✅ Mobil responsive

### Test Listesi:
- ✅ Ana sayfa açılıyor
- ✅ Demo data yükleniyor  
- ✅ Dosya upload çalışıyor
- ✅ Grafik oluşturuluyor
- ✅ Mobil erişim çalışıyor

### Güncelleme Süreci:
```bash
# Kod değişikliği sonrası
git add .
git commit -m "Update description"
git push origin master
# → https://taxoconserv.streamlit.app otomatik güncellenir
```

### Monitoring:
- **Streamlit Cloud Dashboard:** https://share.streamlit.io/
- **App logs** ve **analytics** mevcut
- **Custom domain** (opsiyonel) ayarlanabilir
