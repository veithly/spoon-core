FROM shell-ai:base

WORKDIR /app
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8000"]