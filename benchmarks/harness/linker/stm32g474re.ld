/* Linker script for STM32G474RE (NUCLEO-G474RE)
 *
 * Memory map:
 *   Flash: 512 KB @ 0x08000000
 *   SRAM1: 80 KB  @ 0x20000000
 *   SRAM2: 16 KB  @ 0x20014000
 *   CCM:   32 KB  @ 0x10000000
 *   Total usable SRAM: 128 KB (SRAM1 + SRAM2 + CCM)
 *
 * For benchmarking, we use SRAM1+SRAM2 contiguously (96 KB) for .bss/.data,
 * and CCM (32 KB) for the stack. The arena (model activations) goes in .bss.
 */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 96K   /* SRAM1 (80K) + SRAM2 (16K) */
    CCM   (rwx) : ORIGIN = 0x10000000, LENGTH = 32K   /* CCM SRAM (data + stack) */
}

/* Entry point */
ENTRY(Reset_Handler)

/* Stack at top of CCM */
_estack = ORIGIN(CCM) + LENGTH(CCM);

SECTIONS
{
    /* Vector table + code + read-only data in flash */
    .isr_vector :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        . = ALIGN(4);
    } > FLASH

    .text :
    {
        . = ALIGN(4);
        *(.text)
        *(.text*)
        *(.rodata)
        *(.rodata*)
        *(.glue_7)
        *(.glue_7t)
        KEEP(*(.init))
        KEEP(*(.fini))
        . = ALIGN(4);
        _etext = .;
    } > FLASH

    /* ARM exception handling */
    .ARM.extab :
    {
        *(.ARM.extab* .gnu.linkonce.armextab.*)
    } > FLASH

    .ARM :
    {
        __exidx_start = .;
        *(.ARM.exidx*)
        __exidx_end = .;
    } > FLASH

    /* Initialized data — stored in flash, copied to SRAM at startup */
    _sidata = LOADADDR(.data);

    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > SRAM AT> FLASH

    /* Uninitialized data (arena, buffers) */
    .bss :
    {
        . = ALIGN(4);
        _sbss = .;
        __bss_start__ = _sbss;
        *(.bss)
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
        __bss_end__ = _ebss;
    } > SRAM

    /* Heap (between end of bss and stack) — not used in benchmark */
    ._user_heap_stack :
    {
        . = ALIGN(8);
        PROVIDE(end = .);
        PROVIDE(_end = .);
    } > SRAM

    /* Benchmark scratch data in CCM (input_buf, cycle_counts) */
    .ccm_bss (NOLOAD) :
    {
        . = ALIGN(4);
        *(.ccm_bss)
        *(.ccm_bss*)
        . = ALIGN(4);
    } > CCM

    /* Stack in CCM (after ccm_bss) */
    .stack (NOLOAD) :
    {
        . = ALIGN(8);
        . = . + 4K;  /* 4 KB stack (sufficient for benchmark) */
        . = ALIGN(8);
    } > CCM

    /* Discard unwanted sections */
    /DISCARD/ :
    {
        libc.a(*)
        libm.a(*)
        libgcc.a(*)
    }
}

/* Provide heap/stack symbols for newlib stubs */
PROVIDE(_Min_Heap_Size = 0);
PROVIDE(_Min_Stack_Size = 0x1000);  /* 4 KB */
