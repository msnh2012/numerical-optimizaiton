
file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(NewtonLM ${CPPS})

target_include_directories(NewtonLM PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(NewtonLM PUBLIC Msnhnet)