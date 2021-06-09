
file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(GuassNewtonSmallJac ${CPPS})

target_include_directories(GuassNewtonSmallJac PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(GuassNewtonSmallJac PUBLIC Msnhnet)