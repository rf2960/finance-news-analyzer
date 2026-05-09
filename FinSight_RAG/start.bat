@echo off
echo ============================================================
echo   FinSight RAG - Starting Web App
echo ============================================================
echo.
cd /d "%~dp0"
python -m streamlit run app.py --server.port 8501
pause
