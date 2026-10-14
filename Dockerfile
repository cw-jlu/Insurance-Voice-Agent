FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
COPY requirements-core.lock requirements-demo.txt ./
RUN pip install --no-cache-dir -r requirements-demo.txt
COPY src ./src
COPY data ./data
COPY web ./web
COPY releases ./releases
COPY scripts/run_demo.py ./scripts/run_demo.py
COPY experiments/results/exp44_voice_current/voice ./experiments/results/exp44_voice_current/voice
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health',timeout=3)" || exit 1
CMD ["python", "scripts/run_demo.py", "--host", "0.0.0.0", "--port", "8080"]
