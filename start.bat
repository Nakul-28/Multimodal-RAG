@echo off
set "PROJECT_ROOT=%~dp0"

echo Starting RAG Pipeline...
echo.

:: Start Ollama in a new window
start "Ollama" cmd /k "ollama serve"

:: Start backend in a new window
start "RAG Backend" cmd /k "cd /d ""%PROJECT_ROOT%backend"" && uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

:: Small delay to let backend start first
timeout /t 2 /nobreak >nul

:: Start frontend in a new window
start "RAG Frontend" cmd /k "cd /d ""%PROJECT_ROOT%frontend"" && npm.cmd run dev"
