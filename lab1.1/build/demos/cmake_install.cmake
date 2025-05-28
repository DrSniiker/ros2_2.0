# Install script for directory: /home/rosdev/ros2_ws_lab/lab1.1/src/demos

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/rosdev/ros2_ws_lab/lab1.1/install/demos")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos/environment" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_environment_hooks/pythonpath.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos/environment" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_environment_hooks/pythonpath.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/src-0.0.1-py3.10.egg-info" TYPE DIRECTORY FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_python/src/src.egg-info/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/src" TYPE DIRECTORY FILES "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/" REGEX "/[^/]*\\.pyc$" EXCLUDE REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(
        COMMAND
        "/usr/bin/python3" "-m" "compileall"
        "/home/rosdev/ros2_ws_lab/lab1.1/install/demos/local/lib/python3.10/dist-packages/src"
      )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos" TYPE DIRECTORY FILES "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/config")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos" TYPE DIRECTORY FILES
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/services"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/topics"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/parameters"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/actions"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/demos" TYPE PROGRAM FILES
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/services/service_client.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/services/service_server.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/topics/listener.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/topics/listener_test.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/topics/listener_niklas.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/topics/listener_emil.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/topics/listener_jonas.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/topics/talker_micke.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/topics/talker_martin.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/topics/talker.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/topics/talker_test.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/parameters/param_talker.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/parameters/config_reader.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/actions/action_client.py"
    "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/src/actions/action_server.py"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/demos")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/demos")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos/environment" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos/environment" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_environment_hooks/path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_environment_hooks/local_setup.bash")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_environment_hooks/local_setup.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_environment_hooks/package.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_index/share/ament_index/resource_index/packages/demos")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos/cmake" TYPE FILE FILES
    "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_core/demosConfig.cmake"
    "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/ament_cmake_core/demosConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/demos" TYPE FILE FILES "/home/rosdev/ros2_ws_lab/lab1.1/src/demos/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/rosdev/ros2_ws_lab/lab1.1/build/demos/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
