CC ?= cc
AR ?= ar
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -pedantic -I. -Iinclude -Ithird_party/unity/src
TEST_CFLAGS ?= $(CFLAGS) -Wno-unused-function
LDFLAGS ?=

BUILD_DIR := build
SRC_DIR := src
TEST_DIR := tests
UNITY_DIR := third_party/unity/src

LIB := $(BUILD_DIR)/libsensor_fusion.a

CORE_SRCS := \
	$(SRC_DIR)/sensor_fusion.c \
	$(SRC_DIR)/sf_align.c \
	$(SRC_DIR)/sf_stationary_mount.c \
	$(SRC_DIR)/sf_ekf.c

SRCS := $(CORE_SRCS)
OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRCS))

UNITY_SRC := $(UNITY_DIR)/unity.c
TEST_SRCS := $(TEST_DIR)/test_main.c
TEST_SUPPORT_SRCS := ekf.c
TEST_BIN := $(BUILD_DIR)/test_sensor_fusion

.PHONY: all clean test

all: $(LIB)

$(LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(AR) crus $@ $^

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_BIN): $(LIB) $(TEST_SRCS) $(TEST_SUPPORT_SRCS) $(UNITY_SRC)
	@mkdir -p $(dir $@)
	$(CC) $(TEST_CFLAGS) $(TEST_SRCS) $(TEST_SUPPORT_SRCS) $(UNITY_SRC) $(LIB) -lm -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)
