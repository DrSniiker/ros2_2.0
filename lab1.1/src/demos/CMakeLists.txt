cmake_minimum_required(VERSION 3.5)
project(demos)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

# Install Python modules
ament_python_install_package(src)

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

# Install Python script directories for access as imports
install(DIRECTORY
  src/services
  src/topics
  src/parameters
  src/actions
  DESTINATION share/${PROJECT_NAME}
)

# Install Python executables
install(PROGRAMS
  src/services/service_client.py
  src/services/service_server.py
  src/topics/listener.py
  src/topics/listener_test.py
  src/topics/listener_niklas.py
  src/topics/listener_emil.py
  src/topics/listener_jonas.py
  src/topics/talker_micke.py
  src/topics/talker_martin.py
  src/topics/talker.py
  src/topics/talker_test.py
  src/parameters/param_talker.py
  src/parameters/config_reader.py
  src/actions/action_client.py
  src/actions/action_server.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
