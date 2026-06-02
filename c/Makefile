CC ?= cc
AR ?= ar

CFLAGS ?= -std=c99 -Wall -Wextra -Werror -pedantic -O2
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?= -lm

BUILD_DIR := build
SF_SRC_DIR := sensor_fusion/src
SF_INC_DIR := sensor_fusion/include
RE_SRC_DIR := road_events/src
RE_INC_DIR := road_events/include
UNITY_DIR := third_party/unity
TEST_DIR := tests

LIB_NAME := libimu_gnss_fusion_c.a
LIB_TARGET := $(BUILD_DIR)/$(LIB_NAME)

LIB_SRCS := \
	$(SF_SRC_DIR)/align.c \
	$(SF_SRC_DIR)/sensor_fusion.c \
	$(SF_SRC_DIR)/ekf/runtime.c \
	$(SF_SRC_DIR)/ekf/generated_model.c \
	$(RE_SRC_DIR)/road_events.c
LIB_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(LIB_SRCS))
UNITY_SRCS := $(UNITY_DIR)/unity.c
UNITY_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(UNITY_SRCS))
TEST_SRCS := $(sort $(wildcard $(TEST_DIR)/*_test.c) $(wildcard $(TEST_DIR)/*_tests.c))
TEST_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))
TEST_TARGETS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/$(TEST_DIR)/%,$(TEST_SRCS))

INCLUDES := -I$(SF_INC_DIR) -I$(SF_SRC_DIR) -I$(SF_SRC_DIR)/ekf -I$(RE_INC_DIR) -I$(UNITY_DIR)

.PHONY: all lib test clean

all: lib test

lib: $(LIB_TARGET)

$(LIB_TARGET): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

test: $(TEST_TARGETS)
	@for test_bin in $(TEST_TARGETS); do ./$$test_bin; done

$(BUILD_DIR)/$(TEST_DIR)/%: $(BUILD_DIR)/$(TEST_DIR)/%.o $(UNITY_OBJS) $(LIB_TARGET)
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
