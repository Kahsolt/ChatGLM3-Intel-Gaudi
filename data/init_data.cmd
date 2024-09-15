@ECHO OFF

PUSHD %~dp0

ECHO Download train data...

IF NOT EXIST AdvertiseGen.tar.gz (
  curl -O https://cloud.tsinghua.edu.cn/seafhttp/files/bf579718-d509-4e2c-a0ef-8bfce46aabf7/AdvertiseGen.tar.gz
)

POPD

ECHO Done!
ECHO.

PAUSE
