message("Jeee, CNTK!")

#--------------------------------------------
# Generating the CNTK project
#--------------------------------------------

# Define executable target
set(CORE "${CMAKE_CURRENT_LIST_DIR}/..")
include_directories(${CNTK_INCLUDE_DIR} ${CMAKE_BINARY_DIR} ${CORE})

add_executable(cppcore nnsharp.h implementation_cntk/cppcore.cpp)

target_link_libraries(cppcore ${CNTK_LIB})

# Copy CNTK DLLs to output folder on Windows
if(WIN32)
    foreach(DLL ${CNTK_DLL})
        add_custom_command(TARGET cppcore POST_BUILD COMMAND
            ${CMAKE_COMMAND} -E copy_if_different ${DLL} $<TARGET_FILE_DIR:cppcore>)
    endforeach()
endif()
