@echo off
REM ============================================
REM  Road Damage Detector - one-click launcher
REM ============================================
call C:\Users\janan.pc\anaconda3\Scripts\activate.bat rdd
cd /d E:\RoadDamageDetector-master
echo Starting Road Damage Detector web app...
streamlit run scripts\app.py
pause
