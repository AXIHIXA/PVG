cmake_minimum_required(VERSION 3.0)
project(PVG)
set(CMAKE_CXX_STANDARD 11)

# These 3 options for Qt support
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

set(FADE2D
        fade2d/include/Bbox2.h
        fade2d/include/Circle2.h
        fade2d/include/Color.h
        fade2d/include/common.h
        fade2d/include/ConstraintGraph2.h
        fade2d/include/ConstraintSegment2.h
        fade2d/include/Edge2.h
        fade2d/include/Fade_2D.h
        fade2d/include/fadeVersion.h
        fade2d/include/freeFunctions.h
        fade2d/include/Label.h
        fade2d/include/MeshGenParams.h
        fade2d/include/MsgBase.h
        fade2d/include/Performance.h
        fade2d/include/Point2.h
        fade2d/include/Segment2.h
        fade2d/include/SegmentChecker.h
        fade2d/include/testDataGenerators.h
        fade2d/include/Triangle2.h
        fade2d/include/TriangleAroundVertexIterator.h
        fade2d/include/UserPredicates.h
        fade2d/include/Vector2.h
        fade2d/include/Visualizer2.h
        fade2d/include/Zone2.h
        )

set(FRONTEND
        frontend/include/Geometry.hpp
        frontend/include/ParseColor.hpp
        frontend/include/pathstroke_auxiliary.h
        frontend/include/psimpl.h
        frontend/include/PVG.h
        frontend/include/RegionFZ.h
        frontend/include/SamplSimpl.hpp
        frontend/include/Strokes.h
        frontend/src/pathstroke_auxiliary.cpp
        frontend/src/RegionFZ.cpp
        frontend/src/Strokes.cpp
        frontend/src/PVG.cpp
        )

set(SOLVER
        solver/include/adaptive_evaluation.h
        solver/include/adaptive_solver.h
        solver/include/auxiliary.h
        solver/include/logger.h
        solver/include/point_vector.h
        solver/include/poisson_solver.h
        solver/include/quadtree.h
        solver/include/region.h
        solver/include/structure.h
        solver/include/tree.hh
        solver/src/adaptive_evaluation.cpp
        solver/src/adaptive_solver.cpp
        solver/src/auxiliary.cpp
        solver/src/logger.cpp
        solver/src/poisson_solver.cpp
        solver/src/quadtree.cpp
        solver/src/region.cpp
        )

set(TINYXML
        tinyxml/include/tinyxml2.h
        tinyxml/src/tinyxml2.cpp
        )

set(SOURCE_FILES
        ${ALGLIB}
        ${FADE2D}
        ${FRONTEND}
        ${SOLVER}
        ${TINYXML}
        main/include/debug.h
        main/include/global.h
        main/src/main.cpp
        )

set(ALL_INCLUDE_DIRS
        alglib/include
        fade2d/include
        frontend/include
        main/include
        solver/include
        tinyxml/include
        )

set(FADE2D_LIBS
        ${CMAKE_SOURCE_DIR}/fade2d/lib/libfade2d.so
        ${CMAKE_SOURCE_DIR}/fade2d/lib/libfade25d.so
        )

set(TBB_LIBS
        /usr/lib/x86_64-linux-gnu/libtbb.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so
        )

set(ALL_LIBS
        ${Boost_LIBRARIES}
        ${FADE2D_LIBS}
        ${OpenCV_LIBS}
        ${TBB_LIBS}
        Eigen3::Eigen
        pthread
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        )

add_definitions(-DST_COLOR -DST_VERBOSE)
#add_definitions(-DSAVE_DECOMPRESSED_PVG_XML)
add_definitions(-DoutFZ)
#add_definitions(-DQUADTREE_VORONOI_OUTPUT)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS})
