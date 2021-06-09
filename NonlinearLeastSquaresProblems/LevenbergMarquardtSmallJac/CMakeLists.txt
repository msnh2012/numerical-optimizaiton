
file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(LevenbergMarquardtSmallJac ${CPPS})

target_include_directories(LevenbergMarquardtSmallJac PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(LevenbergMarquardtSmallJac PUBLIC Msnhnet)