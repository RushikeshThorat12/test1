@echo off
REM PDF Compressor Setup Script for Windows

echo.
echo ================================
echo PDF Crush Setup
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [1/3] Python version:
python --version
echo.

echo [2/3] Installing Python dependencies...
pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

echo [3/3] Starting PDF Crush...
echo.
echo ================================
echo Server starting...
echo Open your browser to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo ================================
echo.

python app.py
pause
