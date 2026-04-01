cmake_minimum_required(VERSION 3.16)

find_program(SF_XTENSA_AR xtensa-esp32s3-elf-ar)
find_program(SF_XTENSA_RANLIB xtensa-esp32s3-elf-ranlib)

if(SF_XTENSA_AR)
    set(CMAKE_AR "${SF_XTENSA_AR}" CACHE FILEPATH "Xtensa archive tool" FORCE)
endif()

if(SF_XTENSA_RANLIB)
    set(CMAKE_RANLIB "${SF_XTENSA_RANLIB}" CACHE FILEPATH "Xtensa ranlib tool" FORCE)
endif()

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32s3)
