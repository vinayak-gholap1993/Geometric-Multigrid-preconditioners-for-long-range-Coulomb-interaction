#src/CMakeLists.txt

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

# set up shared library by hand:
ADD_LIBRARY(step-50 SHARED step-50.cc)
DEAL_II_SETUP_TARGET(step-50)

# set up executable with autopilot macro:
SET(TARGET "main")
SET(TARGET_SRC main.cc)
DEAL_II_INVOKE_AUTOPILOT()
TARGET_LINK_LIBRARIES(${TARGET} step-50)
