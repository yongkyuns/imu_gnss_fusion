CC ?= cc
AR ?= ar
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -pedantic -I. -Iinclude -Ithird_party/unity/src
TEST_CFLAGS ?= $(CFLAGS) -Wno-unused-function
LDFLAGS ?=

EMBEDDED_CC ?= arm-none-eabi-gcc
EMBEDDED_AR ?= arm-none-eabi-ar
EMBEDDED_CFLAGS ?= -std=c11 -Os -Wall -Wextra -Werror \
	-ffreestanding -fno-builtin -ffunction-sections -fdata-sections \
	-fsingle-precision-constant -I. -Iinclude \
	-mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard
EMBEDDED_M7_CFLAGS ?= -std=c11 -Os -Wall -Wextra -Werror \
	-ffreestanding -fno-builtin -ffunction-sections -fdata-sections \
	-fsingle-precision-constant -I. -Iinclude \
	-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard
QEMU_CC ?= arm-none-eabi-gcc
QEMU_CFLAGS ?= -std=c11 -Os -Wall -Wextra -Werror \
	-ffreestanding -fno-builtin -ffunction-sections -fdata-sections \
	-fsingle-precision-constant -I. -Iinclude -Ithird_party/unity/src \
	-mcpu=cortex-m33 -mthumb -mfloat-abi=soft
QEMU_TEST_CFLAGS ?= $(QEMU_CFLAGS) -Wno-unused-function
QEMU_LDFLAGS ?= -Ttests/embedded.ld --specs=nano.specs -nostartfiles \
	-Wl,--gc-sections -Wl,-Map,$(QEMU_BUILD_DIR)/test_sensor_fusion.map
QEMU_LIBS ?= -lm -lc -lgcc
QEMU ?= qemu-system-arm
QEMU_MACHINE ?= mps2-an505
QEMU_RUN_FLAGS ?= -M $(QEMU_MACHINE) -cpu cortex-m33 -nographic \
	-semihosting-config enable=on,target=native
QEMU_M7_CFLAGS ?= -std=c11 -Os -Wall -Wextra -Werror \
	-ffreestanding -fno-builtin -ffunction-sections -fdata-sections \
	-fsingle-precision-constant -I. -Iinclude -Ithird_party/unity/src \
	-mcpu=cortex-m7 -mthumb -mfloat-abi=soft
QEMU_M7_TEST_CFLAGS ?= $(QEMU_M7_CFLAGS) -Wno-unused-function
QEMU_M7_LDFLAGS ?= -Ttests/embedded_m7.ld --specs=nano.specs -nostartfiles \
	-Wl,--gc-sections -Wl,-Map,$(QEMU_M7_BUILD_DIR)/test_sensor_fusion.map
QEMU_M7_LIBS ?= -lm -lc -lgcc
QEMU_M7_MACHINE ?= mps2-an500
QEMU_M7_RUN_FLAGS ?= -M $(QEMU_M7_MACHINE) -cpu cortex-m7 -nographic \
	-semihosting-config enable=on,target=native
QEMU_RUNNER ?= python3 tests/run_with_timeout.py
QEMU_TIMEOUT_S ?= 60

BUILD_DIR := build
SRC_DIR := src
TEST_DIR := tests
UNITY_DIR := third_party/unity/src

EMBEDDED_BUILD_DIR := $(BUILD_DIR)/stm32h5
EMBEDDED_M7_BUILD_DIR := $(BUILD_DIR)/stm32h7
QEMU_BUILD_DIR := $(BUILD_DIR)/qemu-stm32h5
QEMU_M7_BUILD_DIR := $(BUILD_DIR)/qemu-stm32h7

LIB := $(BUILD_DIR)/libsensor_fusion.a
EMBEDDED_LIB := $(EMBEDDED_BUILD_DIR)/libsensor_fusion.a
EMBEDDED_M7_LIB := $(EMBEDDED_M7_BUILD_DIR)/libsensor_fusion.a
QEMU_TEST_BIN := $(QEMU_BUILD_DIR)/test_sensor_fusion.elf
QEMU_M7_TEST_BIN := $(QEMU_M7_BUILD_DIR)/test_sensor_fusion.elf

CORE_SRCS := \
	$(SRC_DIR)/sensor_fusion.c \
	$(SRC_DIR)/sf_align.c \
	$(SRC_DIR)/sf_stationary_mount.c \
	$(SRC_DIR)/sf_ekf.c \
	$(SRC_DIR)/sf_eskf.c

SRCS := $(CORE_SRCS)
OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRCS))
EMBEDDED_OBJS := $(patsubst %.c,$(EMBEDDED_BUILD_DIR)/%.o,$(SRCS))
EMBEDDED_M7_OBJS := $(patsubst %.c,$(EMBEDDED_M7_BUILD_DIR)/%.o,$(SRCS))
QEMU_OBJS := $(patsubst %.c,$(QEMU_BUILD_DIR)/%.o,$(SRCS))
QEMU_M7_OBJS := $(patsubst %.c,$(QEMU_M7_BUILD_DIR)/%.o,$(SRCS))

UNITY_SRC := $(UNITY_DIR)/unity.c
TEST_SRCS := $(TEST_DIR)/test_main.c
TEST_SUPPORT_SRCS := ekf.c
TEST_BIN := $(BUILD_DIR)/test_sensor_fusion
QEMU_LD_SCRIPT := $(TEST_DIR)/embedded.ld
QEMU_M7_LD_SCRIPT := $(TEST_DIR)/embedded_m7.ld
QEMU_TEST_SUPPORT_SRCS := $(TEST_SUPPORT_SRCS) $(UNITY_SRC) $(TEST_SRCS) \
	$(TEST_DIR)/embedded_startup.c $(TEST_DIR)/embedded_semihosting.c

.PHONY: all clean test linux-gcc stm32h5 stm32h7 qemu-stm32h5 qemu-stm32h7

all: $(LIB)

linux-gcc:
	$(MAKE) clean
	$(MAKE) all CC=gcc AR=ar

stm32h5: $(EMBEDDED_LIB)

stm32h7: $(EMBEDDED_M7_LIB)

qemu-stm32h5: $(QEMU_TEST_BIN)
	$(QEMU_RUNNER) $(QEMU_TIMEOUT_S) $(QEMU) $(QEMU_RUN_FLAGS) -kernel $(QEMU_TEST_BIN)

qemu-stm32h7: $(QEMU_M7_TEST_BIN)
	$(QEMU_RUNNER) $(QEMU_TIMEOUT_S) $(QEMU) $(QEMU_M7_RUN_FLAGS) -kernel $(QEMU_M7_TEST_BIN)

$(EMBEDDED_BUILD_DIR)/src:
	mkdir -p $@

$(QEMU_BUILD_DIR)/src:
	mkdir -p $@

$(EMBEDDED_M7_BUILD_DIR)/src:
	mkdir -p $@

$(QEMU_BUILD_DIR)/tests:
	mkdir -p $@

$(QEMU_M7_BUILD_DIR)/src:
	mkdir -p $@

$(QEMU_M7_BUILD_DIR)/tests:
	mkdir -p $@

$(QEMU_BUILD_DIR)/third_party/unity/src:
	mkdir -p $@

$(QEMU_M7_BUILD_DIR)/third_party/unity/src:
	mkdir -p $@

$(LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(AR) crus $@ $^

$(EMBEDDED_LIB): $(EMBEDDED_OBJS)
	@mkdir -p $(dir $@)
	$(EMBEDDED_AR) crus $@ $^

$(EMBEDDED_M7_LIB): $(EMBEDDED_M7_OBJS)
	@mkdir -p $(dir $@)
	$(EMBEDDED_AR) crus $@ $^

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(EMBEDDED_BUILD_DIR)/%.o: %.c | $(EMBEDDED_BUILD_DIR)/src
	@mkdir -p $(dir $@)
	$(EMBEDDED_CC) $(EMBEDDED_CFLAGS) -c $< -o $@

$(EMBEDDED_M7_BUILD_DIR)/%.o: %.c | $(EMBEDDED_M7_BUILD_DIR)/src
	@mkdir -p $(dir $@)
	$(EMBEDDED_CC) $(EMBEDDED_M7_CFLAGS) -c $< -o $@

$(QEMU_BUILD_DIR)/%.o: %.c | $(QEMU_BUILD_DIR)/src $(QEMU_BUILD_DIR)/tests $(QEMU_BUILD_DIR)/third_party/unity/src
	@mkdir -p $(dir $@)
	$(QEMU_CC) $(QEMU_TEST_CFLAGS) -c $< -o $@

$(QEMU_M7_BUILD_DIR)/%.o: %.c | $(QEMU_M7_BUILD_DIR)/src $(QEMU_M7_BUILD_DIR)/tests $(QEMU_M7_BUILD_DIR)/third_party/unity/src
	@mkdir -p $(dir $@)
	$(QEMU_CC) $(QEMU_M7_TEST_CFLAGS) -c $< -o $@

$(TEST_BIN): $(LIB) $(TEST_SRCS) $(TEST_SUPPORT_SRCS) $(UNITY_SRC)
	@mkdir -p $(dir $@)
	$(CC) $(TEST_CFLAGS) $(TEST_SRCS) $(TEST_SUPPORT_SRCS) $(UNITY_SRC) $(LIB) -lm -o $@

$(QEMU_TEST_BIN): $(QEMU_OBJS) $(patsubst %.c,$(QEMU_BUILD_DIR)/%.o,$(QEMU_TEST_SUPPORT_SRCS)) $(QEMU_LD_SCRIPT)
	@mkdir -p $(dir $@)
	$(QEMU_CC) $(QEMU_CFLAGS) $(QEMU_LDFLAGS) $(filter %.o,$^) $(QEMU_LIBS) -o $@

$(QEMU_M7_TEST_BIN): $(QEMU_M7_OBJS) $(patsubst %.c,$(QEMU_M7_BUILD_DIR)/%.o,$(QEMU_TEST_SUPPORT_SRCS)) $(QEMU_M7_LD_SCRIPT)
	@mkdir -p $(dir $@)
	$(QEMU_CC) $(QEMU_M7_CFLAGS) $(QEMU_M7_LDFLAGS) $(filter %.o,$^) $(QEMU_M7_LIBS) -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)
