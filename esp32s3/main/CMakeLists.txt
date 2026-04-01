idf_component_register(
    SRCS "app_main.c"
    INCLUDE_DIRS "."
    REQUIRES sensor_fusion esp_driver_usb_serial_jtag esp_timer
)
