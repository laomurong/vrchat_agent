@echo off
rem ===========================================================================
rem One-click launcher for CosyVoice 2.0-0.5B voice-clone TTS service (Task 1)
rem Double-click this file (equivalent to running start_cosyvoice.ps1)
rem NOTE: keep this file ASCII-only (cmd.exe ANSI parsing; Chinese text lives
rem       in the .ps1 which is UTF-8 with BOM).
rem ===========================================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_cosyvoice.ps1" %*
