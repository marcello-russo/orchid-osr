#!/bin/bash
set -e

# Dataset and models are now downloaded during Docker build phase
# via backend/scripts/download_cache.py

exec python server.py
