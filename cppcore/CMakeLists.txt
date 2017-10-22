cmake_minimum_required(VERSION 3.9.2)
project(cppcore)

message("Start generating!")


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/thirdparty")
if(BACKEND STREQUAL "tensorflow")
   
   find_package(TF REQUIRED)
   include("implementation_tf/GenTF.cmake")

elseif(BACKEND STREQUAL "cntk")
   
   find_package(CNTK REQUIRED)
   include("implementation_cntk/GenCNTK.cmake")

else()

   message("Unknown backand!")

endif()

