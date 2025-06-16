#!/bin/bash
set -e

echo "🔄 Installing system dependencies..."
apt-get update -qq && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi7 \
    libxml2 \
    libxslt1.1 \
    python3.10-dev  # Gunakan Python 3.10 yang lebih stabil

echo "🔧 Setting up Python environment..."
python -m pip install --upgrade pip setuptools wheel
