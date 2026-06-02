#ifndef UNITY_H
#define UNITY_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*UnityTestFunction)(void);

extern unsigned int UnityTestsRun;
extern unsigned int UnityTestsFailed;

void UnityBegin(const char *name);
int UnityEnd(void);
void UnityRunTest(UnityTestFunction func, const char *name, const char *file, int line);
void UnityAssertTrue(bool condition, const char *expr, const char *file, int line);
void UnityAssertEqualInt(long expected, long actual, const char *file, int line);
void UnityAssertEqualUInt(uint32_t expected, uint32_t actual, const char *file, int line);
void UnityAssertFloatWithin(float delta, float expected, float actual, const char *file, int line);

#define UNITY_BEGIN() UnityBegin(__FILE__)
#define UNITY_END() UnityEnd()
#define RUN_TEST(func) UnityRunTest((func), #func, __FILE__, __LINE__)

#define TEST_ASSERT_TRUE(condition) \
    UnityAssertTrue((condition), #condition, __FILE__, __LINE__)
#define TEST_ASSERT_FALSE(condition) \
    UnityAssertTrue(!(condition), "!(" #condition ")", __FILE__, __LINE__)
#define TEST_ASSERT_EQUAL_INT(expected, actual) \
    UnityAssertEqualInt((long)(expected), (long)(actual), __FILE__, __LINE__)
#define TEST_ASSERT_EQUAL_UINT32(expected, actual) \
    UnityAssertEqualUInt((uint32_t)(expected), (uint32_t)(actual), __FILE__, __LINE__)
#define TEST_ASSERT_EQUAL_UINT(expected, actual) \
    UnityAssertEqualUInt((uint32_t)(expected), (uint32_t)(actual), __FILE__, __LINE__)
#define TEST_ASSERT_EQUAL(expected, actual) TEST_ASSERT_EQUAL_INT((expected), (actual))
#define TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual) \
    UnityAssertFloatWithin((float)(delta), (float)(expected), (float)(actual), __FILE__, __LINE__)
#define TEST_ASSERT_NOT_EQUAL(not_expected, actual) \
    UnityAssertTrue((long)(not_expected) != (long)(actual), #actual, __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif
