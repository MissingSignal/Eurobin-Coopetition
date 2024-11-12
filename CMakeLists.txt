cmake_minimum_required(VERSION 3.0.2)
project(eurobin_coopetition)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
  geometry_msgs
)

add_service_files(
  FILES
  PickService.srv
  HappyPoseService.srv
  NavService.srv
  EnableAutoModeService.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  modules/speech2text/speech2text.py
  modules/agent/agent.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)