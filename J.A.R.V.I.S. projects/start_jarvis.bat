@echo off
title J.A.R.V.I.S. AI Training Pipeline
color 0B

echo.
echo    ╔══════════════════════════════════════════════════════════════╗
echo    ║                                                              ║
echo    ║    🤖 J.A.R.V.I.S. - Just A Rather Very Intelligent System  ║
echo    ║                                                              ║
echo    ║    AI Training Pipeline for Limited Hardware Resources       ║
echo    ║                                                              ║
echo    ╚══════════════════════════════════════════════════════════════╝
echo.

echo Welcome to J.A.R.V.I.S. AI Training Pipeline!
echo.

:menu
echo Please select an option:
echo.
echo [1] 🌐 Start Web Interface (Recommended)
echo [2] 🚀 Start Training Pipeline
echo [3] 💬 Chat with J.A.R.V.I.S.
echo [4] 📊 Check Status
echo [5] 🔧 Setup Environment
echo [6] 📝 Run Example Demo
echo [7] 📥 Download Datasets
echo [8] ❌ Exit
echo.

set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto web
if "%choice%"=="2" goto train
if "%choice%"=="3" goto chat
if "%choice%"=="4" goto status
if "%choice%"=="5" goto setup
if "%choice%"=="6" goto demo
if "%choice%"=="7" goto download
if "%choice%"=="8" goto exit
goto menu

:web
echo.
echo 🌐 Starting J.A.R.V.I.S. Web Interface...
echo 📱 Open your browser to: http://localhost:8501
echo ⏹️  Press Ctrl+C to stop
echo.
python start_jarvis.py --mode web
goto menu

:train
echo.
echo 🚀 Starting J.A.R.V.I.S. Training Pipeline...
echo.
python start_jarvis.py --mode train
goto menu

:chat
echo.
echo 💬 Starting J.A.R.V.I.S. Chat Interface...
echo.
python start_jarvis.py --mode chat
goto menu

:status
echo.
echo 📊 J.A.R.V.I.S. Status Report
echo ================================
python start_jarvis.py --mode status
echo.
pause
goto menu

:setup
echo.
echo 🔧 Setting up J.A.R.V.I.S. environment...
python start_jarvis.py --mode setup
echo.
pause
goto menu

:demo
echo.
echo 📝 Running J.A.R.V.I.S. Example Demo...
python example_usage.py
echo.
pause
goto menu

:download
echo.
echo 📥 Downloading J.A.R.V.I.S. Datasets...
python start_jarvis.py --mode download
echo.
pause
goto menu

:exit
echo.
echo Thank you for using J.A.R.V.I.S.!
echo "Sometimes you gotta run before you can walk." - Tony Stark
echo.
pause
exit 