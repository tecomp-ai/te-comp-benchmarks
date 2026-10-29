/* Auto-generated benchmark main for: {model_name}
 *
 * Target board: {board_name}
 * MCU: {mcu_name}
 * Clock: {clock_mhz} MHz
 *
 * Runs {n_runs} inferences on {n_inputs} test inputs, measures DWT cycle
 * counts, and reports min/max/avg latency over UART (or semihosting).
 *
 * Build with:
 *   arm-none-eabi-gcc -mcpu={cpu} -mthumb -mfloat-abi=soft -Os \
 *       -ffunction-sections -fdata-sections -Wl,--gc-sections \
 *       -T {linker_script} \
 *       benchmark_main_{model_name}.c {model_name}_engine.c inference_task.c \
 *       startup.c system.c -o benchmark_{model_name}.elf -lm -lnosys
 */

#include "{model_name}_engine.h"
#include "test_data_{model_name}.h"

#include <stdint.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Configuration
 * ----------------------------------------------------------------------- */

#define N_WARMUP_RUNS   3
#define N_TIMED_RUNS    {n_runs}
#define CLOCK_MHZ       {clock_mhz}

/* -----------------------------------------------------------------------
 * DWT Cycle Counter (Cortex-M3/M4/M7)
 * ----------------------------------------------------------------------- */

#define DWT_CTRL_REG    (*(volatile uint32_t *)0xE0001000)
#define DWT_CYCCNT_REG  (*(volatile uint32_t *)0xE0001004)
#define DEMCR_REG       (*(volatile uint32_t *)0xE000EDFC)

static void dwt_init(void) {{
    DEMCR_REG |= (1UL << 24);   /* Enable DWT */
    DWT_CYCCNT_REG = 0;
    DWT_CTRL_REG |= 1;          /* Enable cycle counter */
}}

static inline void dwt_reset(void) {{
    DWT_CYCCNT_REG = 0;
}}

static inline uint32_t dwt_read(void) {{
    return DWT_CYCCNT_REG;
}}

/* -----------------------------------------------------------------------
 * UART output (minimal, board-specific)
 *
 * Override uart_putchar() for your board. Default uses ARM semihosting.
 * ----------------------------------------------------------------------- */

/* ARM semihosting write character */
static void uart_putchar(char c) {{
    /* Semihosting SYS_WRITEC (0x03) */
    volatile char ch = c;
    __asm volatile (
        "mov r0, #0x03\n"
        "mov r1, %0\n"
        "bkpt #0xAB\n"
        :
        : "r" (&ch)
        : "r0", "r1", "memory"
    );
}}

static void uart_puts(const char *s) {{
    while (*s) uart_putchar(*s++);
}}

static void uart_put_uint32(uint32_t v) {{
    char buf[12];
    int i = 0;
    if (v == 0) {{ uart_putchar('0'); return; }}
    while (v > 0) {{
        buf[i++] = '0' + (v % 10);
        v /= 10;
    }}
    while (i > 0) uart_putchar(buf[--i]);
}}

static void uart_put_int32(int32_t v) {{
    if (v < 0) {{ uart_putchar('-'); v = -v; }}
    uart_put_uint32((uint32_t)v);
}}

static void uart_newline(void) {{
    uart_putchar('\r');
    uart_putchar('\n');
}}

/* -----------------------------------------------------------------------
 * Result storage
 * ----------------------------------------------------------------------- */

/* Place large scratch buffers in CCM on STM32G4 (saves main SRAM for arena).
   On S32K344 / other targets, .ccm_bss maps to regular SRAM (no-op). */
static uint32_t cycle_counts[N_TIMED_RUNS] __attribute__((section(".ccm_bss")));
static int8_t input_buf[TEST_INPUT_SIZE]   __attribute__((section(".ccm_bss")));
static int8_t output_buf[TEST_OUTPUT_SIZE];

/* -----------------------------------------------------------------------
 * Main benchmark
 * ----------------------------------------------------------------------- */

static int find_argmax(const int8_t *data, int size) {{
    int max_idx = 0;
    int8_t max_val = data[0];
    for (int i = 1; i < size; i++) {{
        if (data[i] > max_val) {{
            max_val = data[i];
            max_idx = i;
        }}
    }}
    return max_idx;
}}

int main(void) {{
    dwt_init();

    /* Initialize the inference engine */
    {model_name}_engine_init();

    uart_puts("=== Benchmark: {model_name} ===\r\n");
    uart_puts("Board: {board_name}\r\n");
    uart_puts("Clock: ");
    uart_put_uint32(CLOCK_MHZ);
    uart_puts(" MHz\r\n");
    uart_puts("Input size: ");
    uart_put_uint32(TEST_INPUT_SIZE);
    uart_puts(" bytes\r\n");
    uart_puts("Output classes: ");
    uart_put_uint32(TEST_N_CLASSES);
    uart_puts("\r\n");
    uart_puts("Warmup runs: ");
    uart_put_uint32(N_WARMUP_RUNS);
    uart_puts("\r\n");
    uart_puts("Timed runs: ");
    uart_put_uint32(N_TIMED_RUNS);
    uart_puts("\r\n\r\n");

    /* --- Warmup --- */
    uart_puts("Warming up...\r\n");
    for (int w = 0; w < N_WARMUP_RUNS; w++) {{
        test_generate_input(w % TEST_N_INPUTS, input_buf);
        {model_name}_engine_run(input_buf, output_buf);
    }}

    /* --- Timed runs --- */
    uart_puts("Running timed inferences...\r\n");

    uint32_t total_cycles = 0;
    uint32_t min_cycles = 0xFFFFFFFF;
    uint32_t max_cycles = 0;

    for (int r = 0; r < N_TIMED_RUNS; r++) {{
        /* Generate test input r into SRAM buffer (input 0 copies from flash,
           inputs 1..N-1 use LCG PRNG — not measured in cycle count) */
        test_generate_input(r % TEST_N_INPUTS, input_buf);

        dwt_reset();
        uint32_t start = dwt_read();

        {model_name}_engine_run(input_buf, output_buf);

        uint32_t end = dwt_read();
        uint32_t elapsed = end - start;

        cycle_counts[r] = elapsed;
        total_cycles += elapsed;
        if (elapsed < min_cycles) min_cycles = elapsed;
        if (elapsed > max_cycles) max_cycles = elapsed;

        /* Print per-run result */
        uart_puts("  run ");
        uart_put_uint32(r);
        uart_puts(": ");
        uart_put_uint32(elapsed);
        uart_puts(" cycles (");
        uart_put_uint32(elapsed / CLOCK_MHZ);
        uart_puts(" us) class=");
        uart_put_int32(find_argmax(output_buf, TEST_OUTPUT_SIZE));
        uart_newline();
    }}

    /* --- Summary --- */
    uint32_t avg_cycles = total_cycles / N_TIMED_RUNS;
    uint32_t avg_us = avg_cycles / CLOCK_MHZ;
    uint32_t min_us = min_cycles / CLOCK_MHZ;
    uint32_t max_us = max_cycles / CLOCK_MHZ;

    uart_puts("\r\n=== Results ===\r\n");
    uart_puts("Model: {model_name}\r\n");
    uart_puts("Runs: ");
    uart_put_uint32(N_TIMED_RUNS);
    uart_newline();

    uart_puts("Min: ");
    uart_put_uint32(min_cycles);
    uart_puts(" cycles (");
    uart_put_uint32(min_us);
    uart_puts(" us)\r\n");

    uart_puts("Max: ");
    uart_put_uint32(max_cycles);
    uart_puts(" cycles (");
    uart_put_uint32(max_us);
    uart_puts(" us)\r\n");

    uart_puts("Avg: ");
    uart_put_uint32(avg_cycles);
    uart_puts(" cycles (");
    uart_put_uint32(avg_us);
    uart_puts(" us)\r\n");

    uart_puts("Avg ms: ");
    uart_put_uint32(avg_us / 1000);
    uart_puts(".");
    uart_put_uint32((avg_us % 1000) / 100);
    uart_puts("\r\n");

    /* Print last output for verification */
    uart_puts("Last output (first 10 values): ");
    for (int i = 0; i < 10 && i < TEST_OUTPUT_SIZE; i++) {{
        uart_put_int32(output_buf[i]);
        uart_putchar(' ');
    }}
    uart_newline();

    uart_puts("Predicted class: ");
    uart_put_int32(find_argmax(output_buf, TEST_OUTPUT_SIZE));
    uart_newline();

    uart_puts("\r\n=== Done ===\r\n");

    /* Halt */
    while (1) __asm volatile ("wfi");
    return 0;
}}
