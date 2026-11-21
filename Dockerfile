FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir -e . --no-deps
EXPOSE 8000
CMD ["python", "-m", "sentiment_agent", "serve", "--port", "8000"]
