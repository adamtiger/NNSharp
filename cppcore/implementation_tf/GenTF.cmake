message("Jeee, TF!")

#--------------------------------------------
# Generating the TF project
#--------------------------------------------

# Define executable target
set(CORE "${CMAKE_CURRENT_LIST_DIR}/..")
include_directories(${TF_INCLUDE_DIR} ${CMAKE_BINARY_DIR} ${CORE})

add_library(cppcore SHARED nnsharp.h implementation_tf/cppcore.cpp)

target_link_libraries(cppcore ${TF_LIB})

# Copy TF DLLs to output folder on Windows
#if(WIN32)
    #foreach(DLL ${TF_DLL})
        #add_custom_command(TARGET cppcore POST_BUILD COMMAND
            #${CMAKE_COMMAND} -E copy_if_different ${DLL} #$<TARGET_FILE_DIR:cppcore>)
    #endforeach()
#endif()