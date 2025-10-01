@echo off
REM Navigate to project folder
cd /d C:\Users\sonik\Desktop\medical-chatbot-refactored

REM Activate virtual environment
call .\.venv\Scripts\activate.bat

REM Run Streamlit
start streamlit run medibot.py

REM Keep the terminal open
pause
