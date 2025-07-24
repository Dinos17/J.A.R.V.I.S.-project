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
echo [6] ❌ Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto web
if "%choice%"=="2" goto train
if "%choice%"=="3" goto chat
if "%choice%"=="4" goto status
if "%choice%"=="5" goto setup
if "%choice%"=="6" goto exit
goto menu

:web
echo.
echo 🌐 Starting J.A.R.V.I.S. Web Interface...
echo 📱 Open your browser to: http://localhost:8501
echo ⏹️  Press Ctrl+C to stop
echo.
python main.py --mode web
goto menu

:train
echo.
echo 🚀 Starting J.A.R.V.I.S. Training Pipeline...
echo.
python main.py --mode train
goto menu

:chat
echo.
echo 💬 Starting J.A.R.V.I.S. Chat Interface...
echo.
python main.py --mode chat
goto menu

:status
echo.
echo 📊 J.A.R.V.I.S. Status Report
echo ================================
python main.py --mode status
echo.
pause
goto menu

:setup
echo.
echo 🔧 Setting up J.A.R.V.I.S. environment...
python main.py --mode setup
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