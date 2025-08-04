# TaxoConserv Deployment Guide

This guide explains how to deploy TaxoConserv to various cloud platforms.

## Quick Deploy Options

### 🚀 Option 1: Streamlit Cloud (Recommended - FREE)

**Currently deployed at: https://taxoconserv.streamlit.app/**

#### Automatic Deployment
- Changes pushed to `main` branch are automatically deployed
- No additional setup required
- Free tier available
- Perfect for data science applications

#### Manual Setup (if needed)
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select repository: `Bilmem2/TaxoConserv`
4. Branch: `main`
5. Main file: `web_taxoconserv.py`
6. Click "Deploy"

---

### 🌐 Option 2: Heroku (FREE tier discontinued, paid plans available)

#### Prerequisites
- Heroku account
- Heroku CLI installed

#### Steps
```bash
# 1. Login to Heroku
heroku login

# 2. Create Heroku app
heroku create your-app-name

# 3. Deploy
git push heroku main

# 4. Open app
heroku open
```

#### Required Files
- `Procfile` ✅ (already included)
- `runtime.txt` ✅ (already included) 
- `requirements.txt` ✅ (already included)

---

### 🐳 Option 3: Docker Deployment

#### Local Docker Build
```bash
# Build image
docker build -t taxoconserv .

# Run container
docker run -p 8501:8501 taxoconserv
```

#### Docker Compose
```yaml
version: '3.8'
services:
  taxoconserv:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
```

---

### ☁️ Option 4: Cloud Platforms

#### Google Cloud Run
```bash
# Build and deploy
gcloud run deploy taxoconserv \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### AWS Elastic Beanstalk
1. Create `application.py`:
```python
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "web_taxoconserv.py", 
                   "--server.port=8501", "--server.address=0.0.0.0"])
```

2. Deploy with EB CLI:
```bash
eb init
eb create
eb deploy
```

#### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name taxoconserv \
  --image taxoconserv:latest \
  --ports 8501 \
  --ip-address public
```

---

## Environment Configuration

### Environment Variables
```bash
# Optional optimizations
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
PYTHONPATH=/app
```

### Performance Tuning
```bash
# For large datasets
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200
```

---

## Monitoring & Maintenance

### Health Checks
- **Streamlit Cloud**: Built-in monitoring
- **Heroku**: Use New Relic or Papertrail
- **Docker**: Built-in healthcheck endpoint
- **Custom**: Monitor `/_stcore/health`

### Logs
```bash
# Heroku
heroku logs --tail

# Docker
docker logs <container-id>

# Streamlit Cloud
# View logs in dashboard
```

---

## Security Best Practices

### 1. Environment Variables
```bash
# Never commit sensitive data
export DATABASE_URL="your-database-url"
export API_KEY="your-api-key"
```

### 2. HTTPS
- Streamlit Cloud: Automatic HTTPS
- Heroku: Automatic HTTPS
- Custom: Use reverse proxy (nginx + Let's Encrypt)

### 3. Rate Limiting
```python
# Add to web_taxoconserv.py
@st.cache(ttl=60)
def rate_limited_function():
    pass
```

---

## Troubleshooting

### Common Issues

#### 1. Memory Errors
```python
# Solution: Add to web_taxoconserv.py
import streamlit as st
st.set_page_config(
    initial_sidebar_state="collapsed",
    layout="wide"
)
```

#### 2. Module Import Errors
```bash
# Ensure all dependencies in requirements.txt
pip freeze > requirements.txt
```

#### 3. Port Issues
```python
# Use environment port
import os
port = int(os.environ.get("PORT", 8501))
```

### Debug Mode
```bash
# Run locally with debug
streamlit run web_taxoconserv.py --logger.level=debug
```

---

## Cost Comparison

| Platform | Free Tier | Paid Plans | Best For |
|----------|-----------|------------|----------|
| **Streamlit Cloud** | ✅ Unlimited | N/A | Data science apps |
| **Heroku** | ❌ (discontinued) | $7+/month | General web apps |
| **Google Cloud Run** | ✅ Generous | Pay per request | Scalable apps |
| **AWS Elastic Beanstalk** | ✅ 12 months | Pay for resources | Enterprise |
| **Azure Container** | ❌ | Pay per hour | Microsoft ecosystem |

---

## Recommended Deployment

**For TaxoConserv, we recommend Streamlit Cloud because:**
- ✅ **Free** and unlimited for public repositories
- ✅ **Automatic** deployment from GitHub
- ✅ **Optimized** for Streamlit applications
- ✅ **Built-in** monitoring and analytics
- ✅ **No configuration** required
- ✅ **Perfect** for data science tools

**Current deployment: https://taxoconserv.streamlit.app/**

---

## Custom Domain (Optional)

### Streamlit Cloud
1. Upgrade to paid plan
2. Add custom domain in settings
3. Configure DNS CNAME record

### Other Platforms
1. Configure domain in platform settings
2. Update DNS records
3. Enable SSL certificate

---

## Backup & Recovery

### Database Backup
```python
# If using external database
import pandas as pd

def backup_data():
    # Export analysis results
    data.to_csv(f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
```

### Configuration Backup
```bash
# Save environment variables
heroku config > config_backup.txt
```

---

*Last updated: August 2025*
*Version: 1.0*
