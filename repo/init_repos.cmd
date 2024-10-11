@ECHO OFF

PUSHD %~dp0

ECHO Init dependent repos...

IF NOT EXIST optimum-habana (
  git clone https://github.com/Kahsolt/optimum-habana
  PUSHD optimum-habana
  pip install . -e
  POPD
)

IF NOT EXIST Model-References (
  git clone https://github.com/HabanaAI/Model-References
)

POPD

ECHO Done!
ECHO.
