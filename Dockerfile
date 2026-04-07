FROM haskell:9.4

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --break-system-packages -r requirements.txt

COPY . .

EXPOSE 8765
CMD ["sh", "-c", "python3 -m uvicorn server.web_server:app --host 0.0.0.0 --port ${PORT:-8765}"]