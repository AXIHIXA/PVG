cmake_minimum_required(VERSION 3.0)
project(PVG)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

set(ALGLIB
        alglib/include/alglibinternal.h
        alglib/include/alglibmisc.h
        alglib/include/ap.h
        alglib/include/dataanalysis.h
        alglib/include/diffequations.h
        alglib/include/fasttransforms.h
        alglib/include/integration.h
        alglib/include/interpolation.h
        alglib/include/linalg.h
        alglib/include/optimization.h
        alglib/include/solvers.h
        alglib/include/specialfunctions.h
        alglib/include/statistics.h
        alglib/include/stdafx.h
        alglib/src/alglibinternal.cpp
        alglib/src/alglibmisc.cpp
        alglib/src/ap.cpp
        alglib/src/dataanalysis.cpp
        alglib/src/diffequations.cpp
        alglib/src/fasttransforms.cpp
        alglib/src/integration.cpp
        alglib/src/interpolation.cpp
        alglib/src/linalg.cpp
        alglib/src/optimization.cpp
        alglib/src/solvers.cpp
        alglib/src/specialfunctions.cpp
        alglib/src/statistics.cpp
        )

set(FZ
        fz/include/Geometry.hpp
        fz/include/ParseColor.hpp
        fz/include/psimpl.h
        fz/include/PVGBuffer.h
        fz/include/RegionFZ.h
        fz/include/SamplSimpl.hpp
        fz/include/Strokes.h
        fz/src/RegionFZ.cpp
        fz/src/Strokes.cpp
        fz/src/PVGBuffer.cpp
        )

set(TINYXML
        tinyxml/include/tinyxml2.h
        tinyxml/src/tinyxml2.cpp
        )

set(SOURCE_FILES
        ${ALGLIB}
        ${FZ}
        ${TINYXML}
        main/include/auxiliary.h
        main/include/debug.h
        main/include/global.h
        main/src/main.cpp
        )

set(ALL_INCLUDE_DIRS
        alglib/include
        fz/include
        main/include
        tinyxml/include
        )

set(TBB_LIBS
        /usr/lib/x86_64-linux-gnu/libtbb.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so
        )

set(ALL_LIBS
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${TBB_LIBS}
        Eigen3::Eigen
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        )

add_definitions(-DST_COLOR -DST_VERBOSE)
add_definitions(-DSAVE_DECOMPRESSED_PVG_XML)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS})
