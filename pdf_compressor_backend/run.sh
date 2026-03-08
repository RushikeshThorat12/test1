#!/bin/bash

# PDF Compressor Setup Script for Unix/Linux/Mac

echo ""
echo "================================"
echo "PDF Crush Setup"
echo "================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3 from https://www.python.org"
    exit 1
fi

echo "[1/3] Python version:"
python3 --version
echo ""

echo "[2/3] Installing Python dependencies..."
pip3 install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    echo "Try running: pip3 install -r requirements.txt"
    exit 1
fi
echo "Dependencies installed successfully!"
echo ""

echo "[3/3] Starting PDF Crush..."
echo ""
echo "================================"
echo "Server starting..."
echo "Open your browser to: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo "================================"
echo ""

python3 app.py
