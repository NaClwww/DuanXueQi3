cmake_minimum_required(VERSION 3.5)
project(hwdb_native)

set(NCNN_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../ncnn-20260526-harmonyos-vulkan-shared")
set(NCNN_ARCH "${CMAKE_OHOS_ARCH_ABI}")
if(NOT NCNN_ARCH)
    set(NCNN_ARCH "arm64-v8a")
endif()
if(NCNN_ARCH STREQUAL "arm64-v8a")
    set(OHOS_LLVM_ARCH "aarch64-linux-ohos")
elseif(NCNN_ARCH STREQUAL "x86_64")
    set(OHOS_LLVM_ARCH "x86_64-linux-ohos")
else()
    message(FATAL_ERROR "Unsupported HarmonyOS native ABI: ${NCNN_ARCH}")
endif()

add_library(ncnn SHARED IMPORTED)
set(NCNN_REAL_LIBRARY "${NCNN_ROOT}/${NCNN_ARCH}/lib/libncnn.so.1.0.20260526")
set(NCNN_LINK_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/libncnn.so.1")
set(OHOS_OPENMP_LIBRARY "${OHOS_SDK_NATIVE}/llvm/lib/${OHOS_LLVM_ARCH}/libomp.so")
configure_file("${NCNN_REAL_LIBRARY}" "${NCNN_LINK_LIBRARY}" COPYONLY)

set_target_properties(ncnn PROPERTIES
    IMPORTED_LOCATION "${NCNN_LINK_LIBRARY}"
    IMPORTED_SONAME "libncnn.so.1"
    INTERFACE_INCLUDE_DIRECTORIES "${NCNN_ROOT}/${NCNN_ARCH}/include"
)

add_library(openmp SHARED IMPORTED)
set_target_properties(openmp PROPERTIES
    IMPORTED_LOCATION "${OHOS_OPENMP_LIBRARY}"
    IMPORTED_SONAME "libomp.so"
)

add_library(entry SHARED napi_init.cpp)
target_compile_features(entry PRIVATE cxx_std_17)
target_link_libraries(entry PUBLIC ncnn openmp libace_napi.z.so)

add_custom_command(TARGET entry POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NCNN_LINK_LIBRARY}"
        "$<TARGET_FILE_DIR:entry>/libncnn.so.1"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OHOS_OPENMP_LIBRARY}"
        "$<TARGET_FILE_DIR:entry>/libomp.so"
)
