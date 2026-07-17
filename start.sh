#!/bin/bash
set -e

echo "=== Pull ultimo codice ==="
git pull origin main

echo "=== Build frontend ==="
cd frontend
npm ci 2>/dev/null || npm ci
npm run build
cd ..

echo "=== Ricostruisci e avvia stack ==="
sudo docker compose build --no-cache backend
sudo docker compose down -t 0
sudo docker compose up -d

echo "=== Attesa startup ==="
sleep 15
echo "=== URL tunnel ==="
sudo docker compose logs tunnel 2>/dev/null | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1
echo "=== Fatto ==="
