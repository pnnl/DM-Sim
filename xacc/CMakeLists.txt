cmake_minimum_required (VERSION 3.13)

project (xacc-dm-sim LANGUAGES C CXX)
enable_testing()

set (CMAKE_STANDARD_REQUIRED ON)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

set(LIBRARY_NAME xacc-dm-sim)

if (NOT XACC_DIR)
  set(XACC_DIR "$ENV{HOME}/.xacc")
endif()

# Find Xacc
find_package(XACC REQUIRED)


file(GLOB SRC *.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME}
                             PUBLIC .)
target_link_libraries(${LIBRARY_NAME}
                        PUBLIC xacc::xacc xacc::quantum_gate
                      )

set(_bundle_name xacc_dm_sim)
set_target_properties(${LIBRARY_NAME}
                      PROPERTIES COMPILE_DEFINITIONS
                                 US_BUNDLE_NAME=${_bundle_name}
                                 US_BUNDLE_NAME
                                 ${_bundle_name})

usfunctionembedresources(TARGET
                         ${LIBRARY_NAME}
                         WORKING_DIRECTORY
                         ${CMAKE_CURRENT_SOURCE_DIR}
                         FILES
                         manifest.json)
                         
if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "@loader_path/../lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)

find_package (CUDA)
if (CUDA_FOUND)
  add_definitions(-DXACC_HAS_CUDA)
  message(STATUS "CUDA found. Add DM-Sim NVIDIA-based accelerator.")
  add_subdirectory(nvidia_omp)
  target_link_libraries(${LIBRARY_NAME} PRIVATE xacc-dm-sim-nvidia-omp)
endif (CUDA_FOUND)
