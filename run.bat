@echo off
rem ---------------------------------------------------------------------------
rem Minimal pipeline launcher. Prefers the interpreter bundled in this package,
rem so the folder keeps working after being copied to another machine.
rem
rem     run.bat                         all songs -> MusicXML / MIDI / PDF
rem     run.bat victim                  one song
rem     run.bat victim --from separate  start from Demucs separation
rem     run.bat --list-stages           list stage names
rem
rem Why python.exe and not Scripts\*.exe:
rem   In a copied environment the console wrappers under Scripts\ (pip.exe,
rem   demucs.exe, ...) have the *original* absolute install path baked in and
rem   break after a move.  python.exe itself and site-packages locate everything
rem   relative to the running interpreter, so they survive a copy.
rem
rem NOTE: keep this file pure ASCII.  cmd.exe decodes .bat using the OEM code
rem   page (GBK here), so UTF-8 Chinese comments get mangled into commands.
rem   Also avoid parenthesised echo blocks: cmd parses brackets inside them.
rem ---------------------------------------------------------------------------
setlocal

set "HERE=%~dp0"
set "PY=%HERE%python\python.exe"

if exist "%PY%" goto run

where python >nul 2>nul
if errorlevel 1 goto noenv
set "PY=python"
echo [i] Bundled interpreter not found, falling back to system python.

:run
"%PY%" "%HERE%run_pipeline.py" %*
exit /b %errorlevel%

:noenv
echo [!] No usable Python interpreter found.
echo     Either: python bundle_env.py    (copy this machine's env into the package)
echo     Or:     python install_env.py    (build a fresh env with pip)
exit /b 1
