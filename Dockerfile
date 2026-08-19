FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi==0.115.* uvicorn==0.34.* httpx==0.28.*

COPY proxy.py /app/proxy.py

ENV PORT=8080
ENV BACKEND_URL=http://127.0.0.1:8000

EXPOSE 8080

CMD ["python", "proxy.py"]
