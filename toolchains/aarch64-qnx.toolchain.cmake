# Request license and download SDP from QNX Software Center: https://www.qnx.com/products/everywhere/

set(CMAKE_SYSTEM_NAME QNX)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if((NOT DEFINED ENV{QNX_HOST}) OR (NOT DEFINED ENV{QNX_TARGET}))
    if(WIN32)
        message(FATAL_ERROR "Please setup environment in cmd, e.g. call C:\Users\zz\qnx800\qnxsdp-env.bat")
    else()
        message(FATAL_ERROR "Please setup environment in bash, e.g. source /home/zz/qnx800/qnxsdp-env.sh")
    endif()
endif()
set(QNX_HOST "$ENV{QNX_HOST}")
set(QNX_TARGET "$ENV{QNX_TARGET}")

find_program(CMAKE_C_COMPILER   aarch64-unknown-nto-qnx8.0.0-gcc    PATHS "${QNX_HOST}/usr/bin" NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER aarch64-unknown-nto-qnx8.0.0-g++    PATHS "${QNX_HOST}/usr/bin" NO_DEFAULT_PATH)
find_program(CMAKE_AR           aarch64-unknown-nto-qnx8.0.0-ar     PATHS "${QNX_HOST}/usr/bin" NO_DEFAULT_PATH)
find_program(CMAKE_STRIP        aarch64-unknown-nto-qnx8.0.0-strip  PATHS "${QNX_HOST}/usr/bin" NO_DEFAULT_PATH)
find_program(CMAKE_RANLIB       aarch64-unknown-nto-qnx8.0.0-ranlib PATHS "${QNX_HOST}/usr/bin" NO_DEFAULT_PATH)

if(NOT CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_PACKAGE)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif()
