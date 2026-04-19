# 로컬이 3.13이라. 이전 버전이면 거기에 맞춰서 수정 3.11가 가장 안정되었다고들 함
FROM python:3.13-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# Render에서 PORT를 주고, 로컬에서는 10000 기본값
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]