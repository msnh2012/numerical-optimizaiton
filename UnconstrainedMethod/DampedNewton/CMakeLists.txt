
file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(DampedNewton ${CPPS})

target_include_directories(DampedNewton PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(DampedNewton PUBLIC Msnhnet)