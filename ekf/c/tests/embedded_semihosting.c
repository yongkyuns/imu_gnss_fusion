#include <stdint.h>

static int sf_semihost_call(int op, void *arg) {
  register int r0 __asm__("r0") = op;
  register void *r1 __asm__("r1") = arg;
  __asm__ volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
  return r0;
}

int putchar(int ch) {
  char c = (char)ch;
  sf_semihost_call(0x03, &c); /* SYS_WRITEC */
  return ch;
}

void sf_test_exit(int status) {
  uint32_t args[2];
  args[0] = 0x20026u; /* ADP_Stopped_ApplicationExit */
  args[1] = (uint32_t)status;
  sf_semihost_call(0x20, args); /* SYS_EXIT_EXTENDED */
  for (;;) {
  }
}
