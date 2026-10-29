/*===- te_parallel_runtime.c - FreeRTOS dual-core sync ------------------===*/
/*                                                                          */
/* FreeRTOS-based synchronization for dual-core parallel matmul.            */
/* Uses binary semaphores for zero-overhead signaling between cores.         */
/*                                                                          */
/* Design (inspired by cardputer-ai/llm.cpp):                               */
/*   - A persistent worker task is pinned to core 1 at high priority        */
/*   - It blocks on a "go" semaphore, waiting for work                      */
/*   - Core 0's te_parallel_begin() sets op_id and gives the "go" sem      */
/*   - Core 1 runs te_parallel_worker_run(op_id), then gives "done" sem    */
/*   - Core 0's te_parallel_join() takes the "done" sem (blocks until done) */
/*   - No context switch on core 1's critical path (task owns the core)     */
/*                                                                          */
/*===--------------------------------------------------------------------===*/

#include "te_parallel_runtime.h"
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

#include <stddef.h>

/* --- Synchronization state --- */

static SemaphoreHandle_t sem_go   = NULL;  /* Core 0 -> Core 1: start work */
static SemaphoreHandle_t sem_done = NULL;  /* Core 1 -> Core 0: work complete */
static TaskHandle_t worker_task   = NULL;
static volatile int64_t worker_op_id = -1;

/* --- Worker task (runs on core 1) --- */

static void te_worker_task(void *arg) {
    (void)arg;

    for (;;) {
        /* Block until core 0 dispatches an op. */
        xSemaphoreTake(sem_go, portMAX_DELAY);

        /* Execute the dispatched parallel op. */
        int64_t op_id = worker_op_id;
        te_parallel_worker_run(op_id);

        /* Signal completion to core 0. */
        xSemaphoreGive(sem_done);

        /* Yield so other tasks on this core can run (optional). */
        taskYIELD();
    }
}

/* --- Core 0 API (called by generated engine code) --- */

void te_parallel_begin(int64_t num_cores, int64_t op_id) {
    (void)num_cores;

    /* Set the op_id for the worker and wake it. */
    worker_op_id = op_id;
    xSemaphoreGive(sem_go);
}

void te_parallel_join(int64_t num_cores) {
    (void)num_cores;

    /* Wait for the worker to finish. */
    xSemaphoreTake(sem_done, portMAX_DELAY);
}

/* --- Initialization --- */

void te_parallel_init(void) {
    /* Create binary semaphores. */
    sem_go   = xSemaphoreCreateBinary();
    sem_done = xSemaphoreCreateBinary();

    configASSERT(sem_go != NULL);
    configASSERT(sem_done != NULL);

    /* Create the worker task, pinned to the worker core.
     * xTaskCreatePinnedToCore is available on ESP-IDF and some FreeRTOS
     * ports. For standard FreeRTOS, use xTaskCreate instead (no pinning). */
#if defined(ESP_PLATFORM) || defined(CONFIG_FREERTOS_SMP)
    xTaskCreatePinnedToCore(
        te_worker_task,
        "te_worker",
        TE_PARALLEL_WORKER_STACK / sizeof(StackType_t),
        NULL,
        TE_PARALLEL_WORKER_PRIORITY,
        &worker_task,
        TE_PARALLEL_WORKER_CORE
    );
#else
    xTaskCreate(
        te_worker_task,
        "te_worker",
        TE_PARALLEL_WORKER_STACK / sizeof(StackType_t),
        NULL,
        TE_PARALLEL_WORKER_PRIORITY,
        &worker_task
    );
#endif

    configASSERT(worker_task != NULL);
}

void te_parallel_deinit(void) {
    if (worker_task != NULL) {
        vTaskDelete(worker_task);
        worker_task = NULL;
    }
    if (sem_go != NULL) {
        vSemaphoreDelete(sem_go);
        sem_go = NULL;
    }
    if (sem_done != NULL) {
        vSemaphoreDelete(sem_done);
        sem_done = NULL;
    }
}
