set(SF_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../ekf/c)

idf_component_register(
    SRCS
        "${SF_ROOT}/src/sensor_fusion.c"
        "${SF_ROOT}/src/sf_align.c"
        "${SF_ROOT}/src/sf_ekf.c"
        "${SF_ROOT}/src/sf_stationary_mount.c"
    INCLUDE_DIRS
        "${SF_ROOT}"
        "${SF_ROOT}/include"
)

target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)
