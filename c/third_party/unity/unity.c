#include "unity.h"

#include <math.h>
#include <stdio.h>

unsigned int UnityTestsRun = 0u;
unsigned int UnityTestsFailed = 0u;

static bool UnityCurrentTestFailed = false;

void UnityBegin(const char *name)
{
    UnityTestsRun = 0u;
    UnityTestsFailed = 0u;
    printf("Unity test run: %s\n", name != 0 ? name : "(unnamed)");
}

int UnityEnd(void)
{
    printf("%u tests, %u failures\n", UnityTestsRun, UnityTestsFailed);
    return UnityTestsFailed == 0u ? 0 : 1;
}

void UnityRunTest(UnityTestFunction func, const char *name, const char *file, int line)
{
    UnityCurrentTestFailed = false;
    UnityTestsRun += 1u;
    printf("RUN %s (%s:%d)\n", name, file, line);
    func();
    if (UnityCurrentTestFailed) {
        UnityTestsFailed += 1u;
        printf("FAIL %s\n", name);
    } else {
        printf("PASS %s\n", name);
    }
}

void UnityAssertTrue(bool condition, const char *expr, const char *file, int line)
{
    if (!condition) {
        UnityCurrentTestFailed = true;
        printf("%s:%d: assertion failed: %s\n", file, line, expr);
    }
}

void UnityAssertEqualInt(long expected, long actual, const char *file, int line)
{
    if (expected != actual) {
        UnityCurrentTestFailed = true;
        printf("%s:%d: expected %ld, got %ld\n", file, line, expected, actual);
    }
}

void UnityAssertEqualUInt(uint32_t expected, uint32_t actual, const char *file, int line)
{
    if (expected != actual) {
        UnityCurrentTestFailed = true;
        printf("%s:%d: expected %u, got %u\n", file, line, expected, actual);
    }
}

void UnityAssertFloatWithin(float delta, float expected, float actual, const char *file, int line)
{
    if (!isfinite(expected) || !isfinite(actual) || fabsf(expected - actual) > delta) {
        UnityCurrentTestFailed = true;
        printf("%s:%d: expected %.9g +/- %.9g, got %.9g\n", file, line, expected, delta, actual);
    }
}
