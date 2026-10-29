/* Minimal Cortex-M startup for benchmarking.
 *
 * Provides Reset_Handler (copies .data, zeroes .bss, calls main),
 * default interrupt handlers, and vector table.
 *
 * For real deployment, use the vendor-provided startup (STM32CubeIDE
 * or S32DS generates these). This file is only for standalone
 * benchmark builds with arm-none-eabi-gcc.
 */

#include <stdint.h>

/* Linker-provided symbols */
extern uint32_t _sidata;   /* .data init values in flash */
extern uint32_t _sdata;    /* .data start in SRAM */
extern uint32_t _edata;    /* .data end in SRAM */
extern uint32_t _sbss;     /* .bss start */
extern uint32_t _ebss;     /* .bss end */
extern uint32_t _estack;   /* Initial stack pointer */

/* Main function */
extern int main(void);

/* Reset handler */
void Reset_Handler(void) {
    /* Copy .data from flash to SRAM */
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;
    while (dst < &_edata)
        *dst++ = *src++;

    /* Zero .bss */
    dst = &_sbss;
    while (dst < &_ebss)
        *dst++ = 0;

    /* Enable FPU if present (Cortex-M4F / M7) */
    /* CPACR: CP10 and CP11 full access */
    *(volatile uint32_t *)0xE000ED88 |= (0xFUL << 20);
    __asm volatile ("dsb");
    __asm volatile ("isb");

    /* Call main */
    main();

    /* Should not return */
    while (1)
        __asm volatile ("wfi");
}

/* Default handler for all interrupts */
void Default_Handler(void) {
    while (1)
        __asm volatile ("bkpt #0");
}

/* Weak aliases — override in your code if needed */
void NMI_Handler(void)         __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void)   __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void)   __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void)    __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void)  __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void)         __attribute__((weak, alias("Default_Handler")));
void DebugMon_Handler(void)    __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void)      __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void)     __attribute__((weak, alias("Default_Handler")));

/* Vector table — minimum for Cortex-M */
__attribute__((section(".isr_vector"), used))
const void *vector_table[] = {
    &_estack,              /* Initial stack pointer */
    Reset_Handler,         /* Reset */
    NMI_Handler,           /* NMI */
    HardFault_Handler,     /* Hard Fault */
    MemManage_Handler,     /* MPU Fault */
    BusFault_Handler,      /* Bus Fault */
    UsageFault_Handler,    /* Usage Fault */
    0, 0, 0, 0,           /* Reserved */
    SVC_Handler,           /* SVCall */
    DebugMon_Handler,      /* Debug Monitor */
    0,                     /* Reserved */
    PendSV_Handler,        /* PendSV */
    SysTick_Handler,       /* SysTick */
    /* IRQ0-239: all default */
};

/* Provide _exit for newlib */
void _exit(int status) {
    (void)status;
    while (1) __asm volatile ("wfi");
}
