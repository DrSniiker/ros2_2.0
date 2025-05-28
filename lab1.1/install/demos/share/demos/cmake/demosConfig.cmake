# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_demos_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED demos_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(demos_FOUND FALSE)
  elseif(NOT demos_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(demos_FOUND FALSE)
  endif()
  return()
endif()
set(_demos_CONFIG_INCLUDED TRUE)

# output package information
if(NOT demos_FIND_QUIETLY)
  message(STATUS "Found demos: 0.0.1 (${demos_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'demos' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${demos_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(demos_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${demos_DIR}/${_extra}")
endforeach()
