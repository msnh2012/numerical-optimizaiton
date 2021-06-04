
file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(QuasiNewton ${CPPS})

target_include_directories(QuasiNewton PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(QuasiNewton PUBLIC Msnhnet)