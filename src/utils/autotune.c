#include "m4linalg.h"
#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>

static int global_l1_tile = 32;
static int global_l2_tile = 64;

void m4_autotune_init(void) {
    char cpu_brand[256];
    size_t size = sizeof(cpu_brand);
    
    /* Query macOS for the exact CPU model */
    if (sysctlbyname("machdep.cpu.brand_string", &cpu_brand, &size, NULL, 0) == 0) {
        if (strstr(cpu_brand, "M4")) {
            /* Apple M4: Larger unified cache hierarchy */
            global_l1_tile = 64; 
            global_l2_tile = 128;
        } else if (strstr(cpu_brand, "M3") || strstr(cpu_brand, "M2")) {
            /* Apple M2/M3 profiles */
            global_l1_tile = 48;
            global_l2_tile = 96;
        } else {
            /* Apple M1 or fallback */
            global_l1_tile = 32; 
            global_l2_tile = 64;
        }
    }
}

int m4_get_l1_tile_size(void) {
    return global_l1_tile;
}
int m4_get_l2_tile_size(void) {
    return global_l2_tile;
}
