/* Linker script for NXP S32K344 (S32K3X4EVB-T172)
 *
 * Memory map (simplified for benchmarking):
 *   PFlash: 4 MB @ 0x00400000 (Code Flash Block 0-3)
 *   SRAM:   512 KB
 *     SRAM_L: 64 KB  @ 0x1FFE0000 (lower, tightly coupled)
 *     SRAM_U: 384 KB @ 0x20000000 (upper)
 *     SRAM_S: 64 KB  @ 0x20060000 (standby)
 *   For simplicity, model SRAM as contiguous 448 KB (SRAM_L + SRAM_U).
 *
 * S32K344 boots from PFlash with the Boot Header at 0x00400000.
 * In practice, use NXP S32DS to generate the real startup/linker.
 * This script is for size analysis and reference.
 */

MEMORY
{
    PFLASH (rx)  : ORIGIN = 0x00400000, LENGTH = 4M
    SRAM   (rwx) : ORIGIN = 0x1FFE0000, LENGTH = 512K   /* SRAM_L (64K) + SRAM_U (384K) + SRAM_S (64K) */
}

/* Entry point */
ENTRY(Reset_Handler)

/* Stack at top of main SRAM */
_estack = ORIGIN(SRAM) + LENGTH(SRAM);

SECTIONS
{
    /* Vector table + code + read-only data in flash */
    .isr_vector :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        . = ALIGN(4);
    } > PFLASH

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
    } > PFLASH

    .ARM.extab :
    {
        *(.ARM.extab* .gnu.linkonce.armextab.*)
    } > PFLASH

    .ARM :
    {
        __exidx_start = .;
        *(.ARM.exidx*)
        __exidx_end = .;
    } > PFLASH

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
    } > SRAM AT> PFLASH

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

    /* Benchmark scratch (input_buf, cycle_counts) — in main SRAM on S32K */
    .ccm_bss (NOLOAD) :
    {
        . = ALIGN(4);
        *(.ccm_bss)
        *(.ccm_bss*)
        . = ALIGN(4);
    } > SRAM

    ._user_heap_stack :
    {
        . = ALIGN(8);
        PROVIDE(end = .);
        PROVIDE(_end = .);
    } > SRAM

    /DISCARD/ :
    {
        libc.a(*)
        libm.a(*)
        libgcc.a(*)
    }
}

PROVIDE(_Min_Heap_Size = 0);
PROVIDE(_Min_Stack_Size = 0x4000);  /* 16 KB stack */
