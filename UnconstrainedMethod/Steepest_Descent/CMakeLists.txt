
file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(Steepest_Descent ${CPPS})

target_include_directories(Steepest_Descent PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(Steepest_Descent PUBLIC Msnhnet)