/*===- te_parallel_runtime.c - Bare-metal dual-core sync ----------------===*/
/*                                                                          */
/* Hardware-level synchronization for dual-core parallel matmul.            */
/* No OS dependency. Uses:                                                  */
/*   - ARM Cortex-M: __DSB + __SEV / __WFE (low-power wait)                */
/*   - Xtensa (ESP32-S3): spin on volatile (or crosscore_int)              */
/*   - Generic: spin on volatile flag                                       */
/*                                                                          */
/* Usage:                                                                   */
/*   Core 0: calls te_parallel_begin/join (from generated engine code)      */
/*   Core 1: call te_parallel_worker_loop() from its startup code           */
/*                                                                          */
/*===--------------------------------------------------------------------===*/

#include "te_parallel_runtime.h"

/* --- Synchronization state (shared between cores) --- */

/* Worker state machine:
 *   IDLE  -> READY (core 0 sets op_id and transitions to READY)
 *   READY -> BUSY  (core 1 sees READY, transitions to BUSY, starts work)
 *   BUSY  -> DONE  (core 1 finishes, transitions to DONE)
 *   DONE  -> IDLE  (core 0 sees DONE, transitions back to IDLE)
 */
enum {
    WORKER_IDLE  = 0,
    WORKER_READY = 1,
    WORKER_BUSY  = 2,
    WORKER_DONE  = 3,
    WORKER_STOP  = 4,
};

static volatile int32_t worker_state = WORKER_IDLE;
static volatile int64_t worker_op_id = -1;

/* --- Platform-specific barrier/signal primitives --- */

#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
/* ARM Cortex-M4/M7/M33: use SEV/WFE for low-power synchronization.
 * __DSB() ensures memory writes are visible before signaling. */

static inline void mem_barrier(void) {
    __asm volatile ("dsb sy" ::: "memory");
}

static inline void signal_event(void) {
    __asm volatile ("dsb sy" ::: "memory");
    __asm volatile ("sev" ::: "memory");
}

static inline void wait_for_event(void) {
    __asm volatile ("wfe" ::: "memory");
}

#elif defined(__XTENSA__)
/* Xtensa (ESP32-S3): no WFE/SEV equivalent. Use memory barrier + spin.
 * On ESP-IDF, you could use esp_crosscore_int for interrupt-based wake,
 * but that requires IDF headers. This generic version just spins. */

static inline void mem_barrier(void) {
    __asm volatile ("memw" ::: "memory");
}

static inline void signal_event(void) {
    __asm volatile ("memw" ::: "memory");
}

static inline void wait_for_event(void) {
    /* Spin — Xtensa has no low-power wait-for-event. */
    __asm volatile ("nop" ::: "memory");
}

#elif defined(__TRICORE__)
/* TriCore (AURIX): use ISYNC for memory ordering + spin. */

static inline void mem_barrier(void) {
    __asm volatile ("isync" ::: "memory");
}

static inline void signal_event(void) {
    __asm volatile ("isync" ::: "memory");
}

static inline void wait_for_event(void) {
    __asm volatile ("nop" ::: "memory");
}

#else
/* Generic fallback: compiler barrier only. */

static inline void mem_barrier(void) {
    __asm volatile ("" ::: "memory");
}

static inline void signal_event(void) {
    __asm volatile ("" ::: "memory");
}

static inline void wait_for_event(void) {
    __asm volatile ("" ::: "memory");
}

#endif

/* --- Core 0 API (called by generated engine code) --- */

void te_parallel_begin(int64_t num_cores, int64_t op_id) {
    (void)num_cores;  /* Currently always 2 cores. */

    /* Set the op_id for core 1 and signal it to start. */
    worker_op_id = op_id;
    mem_barrier();
    worker_state = WORKER_READY;
    signal_event();
}

void te_parallel_join(int64_t num_cores) {
    (void)num_cores;

    /* Wait for core 1 to finish. */
    while (worker_state != WORKER_DONE) {
        wait_for_event();
    }

    /* Reset to idle. */
    worker_state = WORKER_IDLE;
    mem_barrier();
}

/* --- Core 1 API (call from core 1's entry point) --- */

void te_parallel_worker_loop(void) {
    for (;;) {
        /* Wait for core 0 to dispatch an op. */
        while (worker_state != WORKER_READY) {
            if (worker_state == WORKER_STOP)
                return;
            wait_for_event();
        }

        /* Transition to BUSY and execute. */
        worker_state = WORKER_BUSY;
        mem_barrier();

        int64_t op_id = worker_op_id;
        te_parallel_worker_run(op_id);

        /* Signal completion. */
        mem_barrier();
        worker_state = WORKER_DONE;
        signal_event();
    }
}

void te_parallel_worker_stop(void) {
    worker_state = WORKER_STOP;
    signal_event();
}
