#!/bin/bash
set -e

echo "=== Build frontend ==="
cd frontend
npm ci --silent 2>/dev/null || npm ci
npm run build
cd ..

echo "=== Avvio stack ==="
sudo docker compose up -d

echo "=== Attesa tunnel ==="
sleep 10
sudo docker compose logs tunnel 2>/dev/null | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1
