
file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(LevenbergMarquardt ${CPPS})

target_include_directories(LevenbergMarquardt PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(LevenbergMarquardt PUBLIC Msnhnet)