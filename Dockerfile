# TaxoConserv Cloud Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check for container orchestration
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run streamlit with cloud-optimized settings
CMD ["streamlit", "run", "web_taxoconserv.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
