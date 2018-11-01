# correspondence grouping sample

set(CORR_GRP_TARGET correspondence_grouping)

# dependent files.
file(GLOB_RECURSE CORR_GRP_BIN_SRCS "*.cpp")

# target
add_executable(${CORR_GRP_TARGET} ${CORR_GRP_TARGET})

# include and libraries
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})
target_include_directories(${CORR_GRP_TARGET} PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(${CORR_GRP_TARGET} PRIVATE ${PCL_LIBRARIES})

# output directory
set_target_properties(${CORR_GRP_TARGET} PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")

# deploy client binary
install(TARGETS ${CORR_GRP_TARGET} RUNTIME DESTINATION bin)


