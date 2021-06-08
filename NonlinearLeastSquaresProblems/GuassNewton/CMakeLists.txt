
file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(GuassNewton ${CPPS})

target_include_directories(GuassNewton PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(GuassNewton PUBLIC Msnhnet)