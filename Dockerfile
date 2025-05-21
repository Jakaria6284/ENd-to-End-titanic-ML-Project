FROM python:3.10-slim

WORKDIR /app

COPY Backend ./Backend
COPY ML ./ML
COPY requirement.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "Backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
