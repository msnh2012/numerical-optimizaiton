
file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(Newton ${CPPS})

target_include_directories(Newton PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(Newton PUBLIC Msnhnet)