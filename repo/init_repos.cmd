@ECHO OFF

PUSHD %~dp0

ECHO Init dependent repos...

IF NOT EXIST optimum-habana (
  git clone https://github.com/huggingface/optimum-habana
  PUSHD optimum-habana
  pip install . -e
  POPD
)

POPD

ECHO Done!
ECHO.

PAUSE
