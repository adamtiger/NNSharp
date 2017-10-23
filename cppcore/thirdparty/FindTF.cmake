#-----------------------------------------------------
# Goal: finding the necessary components of tensorflow
# 	   on the current platform.
# 
# Sets variables:
#     TF_INCLUDE_DIR
#     TF_LIB
#     TF_DLL
#-----------------------------------------------------

include(FindPackageHandleStandardArgs)

set(TF_ROOT "${CMAKE_CURRENT_LIST_DIR}/tf")
if(WIN32)
   if (EXISTS "${TF_ROOT}/tensorflow.h")
       set(TF_INCLUDE_DIR "${TF_ROOT}")
       set(TF_LIB "${TF_ROOT}/tensorflow.lib")
       set(TF_DLL "${TF_ROOT}/tensorflow.dll")
   endif()
   mark_as_advanced(TF_ROOT)
   find_package_handle_standard_args(TF DEFAULT_MSG TF_INCLUDE_DIR TF_LIB TF_DLL)

else()
   if (EXISTS "${TF_ROOT}/tensorflow.h")
       set(TF_INCLUDE_DIR "${TF_ROOT}")
       set(TF_LIB "${TF_ROOT}/libtensorflow.so")
   endif()
   mark_as_advanced(TF_ROOT)
   find_package_handle_standard_args(TF DEFAULT_MSG TF_INCLUDE_DIR TF_LIB)

endif()
