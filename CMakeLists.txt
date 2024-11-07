cmake_minimum_required(VERSION 3.0.2)
project(eurobin_coopetion)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

## Generate services in the 'srv' folder
add_service_files(
   FILES
Pick.srv
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  modules/speech2text/speech2text.py
  # src/close_gstreamer_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)