cmake_minimum_required(VERSION 3.5)
project(bringup)
find_package(ament_cmake REQUIRED)
install(DIRECTORY launch models rviz urdf worlds params maps
  DESTINATION share/${PROJECT_NAME}/
)
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/.gazebo.dsv.in")
ament_package()
