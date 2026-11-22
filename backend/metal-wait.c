/* Used only for an explicitly requested, single-stream 9B fast chat.
   Queue is the same queue the pinned GGML Metal device used to submit the pass.
   An empty marker is ordered after that pass without reading backend structs.
   The caller always performs its ordinary sampled-token read afterwards. */
#include <objc/runtime.h>
#include <objc/message.h>
#include <stdint.h>
#include <time.h>
#include "ggml-backend-impl.h"

extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *);

/* The registered MTL0 device is the device ggml_backend_metal_device_init_backend
   uses. Compile against the very same ggml-backend-impl.h as the pinned runtime. */
void *ternzip_registered_device_context(ggml_backend_dev_t device) {
    return device ? device->context : NULL;
}

void ternzip_wait_for_queue(id queue) {
    if (!queue) return;
    void *pool = objc_autoreleasePoolPush();
    id (*object)(id, SEL) = (void *)objc_msgSend;
    void (*action)(id, SEL) = (void *)objc_msgSend;
    unsigned long (*number)(id, SEL) = (void *)objc_msgSend;
    id marker = object(queue, sel_registerName("commandBuffer"));
    if (marker) {
        SEL status = sel_registerName("status");
        action(marker, sel_registerName("commit"));
        uint64_t limit = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) + 100000000ULL;
        while (number(marker, status) < 4) {
            if (clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) >= limit) break;
            for (int i = 0; i < 64; ++i) __asm__ volatile("yield");
        }
    }
    objc_autoreleasePoolPop(pool);
}
