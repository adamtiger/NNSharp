#-----------------------------------------------------
# Goal: finding the necessary components of cntk
# 	   on the current platform.
# 
# Sets variables:
#     CNTK_INCLUDE_DIR
#     CNTK_LIB
#     CNTK_DLL
#-----------------------------------------------------

include(FindPackageHandleStandardArgs)

set(CNTK_ROOT "${CMAKE_CURRENT_LIST_DIR}/cntk")
if(WIN32)
   if (EXISTS "${CNTK_ROOT}/include/CNTKLibrary.h")
       set(CNTK_INCLUDE_DIR "${CNTK_ROOT}/include")
       set(CNTK_LIB "${CNTK_ROOT}/lib/Cntk.Core-2.2.lib")
       set(CNTK_DLL "${CNTK_ROOT}/dll/Cntk.Composite-2.2.dll" "${CNTK_ROOT}/dll/Cntk.Core.CSBinding-2.2.dll" "${CNTK_ROOT}/dll/Cntk.Core-2.2.dll" "${CNTK_ROOT}/dll/Cntk.Deserializers.Binary-2.2.dll" "${CNTK_ROOT}/dll/Cntk.Deserializers.HTK-2.2.dll" "${CNTK_ROOT}/dll/Cntk.Deserializers.Image-2.2.dll" "${CNTK_ROOT}/dll/Cntk.Deserializers.TextFormat-2.2.dll" "${CNTK_ROOT}/dll/Cntk.Math-2.2.dll" "${CNTK_ROOT}/dll/Cntk.PerformanceProfiler-2.2.dll" "${CNTK_ROOT}/dll/libiomp5md.dll" "${CNTK_ROOT}/dll/mkl_cntk_p.dll" "${CNTK_ROOT}/dll/opencv_world310.dll" "${CNTK_ROOT}/dll/zip.dll" "${CNTK_ROOT}/dll/zlib.dll")
   endif()
   mark_as_advanced(CNTK_ROOT)
   find_package_handle_standard_args(CNTK DEFAULT_MSG CNTK_INCLUDE_DIR CNTK_LIB CNTK_DLL)

else()
   if (EXISTS "${CNTK_ROOT}/include/CNTKLibrary.h")
       set(CNTK_INCLUDE_DIR "${CNTK_ROOT}/include")
       set(CNTK_LIB "${CNTK_ROOT}/lib/libCntk.Core-2.2.so" "${CNTK_ROOT}/lib/libCntk.Eval-2.2.so" "${CNTK_ROOT}/lib/libCntk.Math-2.2.so" "${CNTK_ROOT}/lib/libCntk.PerformanceProfiler-2.2.so" "${CNTK_ROOT}/lib/libfst.so.3" "${CNTK_ROOT}/lib/libiomp5.so" "${CNTK_ROOT}/lib/libkaldi-base.so" "${CNTK_ROOT}/lib/libkaldi-cudamatrix.so" "${CNTK_ROOT}/lib/libkaldi-hmm.so" "${CNTK_ROOT}/lib/libkaldi-lat.so" "${CNTK_ROOT}/lib/libkaldi-matrix.so" "${CNTK_ROOT}/lib/libkaldi-nnet.so" "${CNTK_ROOT}/lib/libkaldi-tree.so" "${CNTK_ROOT}/lib/libkaldi-util.so" "${CNTK_ROOT}/lib/libmkl_cntk_p.so" "${CNTK_ROOT}/lib/libmultiverso.so" "${CNTK_ROOT}/lib/libopenblas.so.0" "${CNTK_ROOT}/lib/libopencv_core.so.3.1" "${CNTK_ROOT}/lib/libopencv_imgproc.so.3.1" "${CNTK_ROOT}/lib/libzip.so.4")
   endif()
   mark_as_advanced(CNTK_ROOT)
   find_package_handle_standard_args(CNTK DEFAULT_MSG CNTK_INCLUDE_DIR CNTK_LIB)

endif()

