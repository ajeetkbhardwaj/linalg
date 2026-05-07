#include "config.h"
#include <time.h>
#include <stdio.h>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

/* Get high-resolution time in seconds */
double get_time_sec(void) {
#if defined(__APPLE__)
    /* Use mach_absolute_time on macOS for nanosecond precision */
    static mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    uint64_t ticks = mach_absolute_time();
    return (ticks * info.numer / info.denom) * 1e-9;
#elif defined(CLOCK_MONOTONIC)
    /* POSIX high-resolution clock */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#else
    /* Fallback to clock() */
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* Benchmark a function call */
typedef void (*bench_func_t)(void *user_data);

double benchmark(bench_func_t func, void *user_data, int iterations) {
    /* Warm-up run */
    func(user_data);
    
    /* Timed runs */
    double t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        func(user_data);
    }
    double t1 = get_time_sec();
    
    return (t1 - t0) / iterations;
}

/* Print performance summary */
void print_perf(const char *label, double time_sec, int flops) {
    double gflops = (flops * 1e-9) / time_sec;
    printf("%-30s: %8.4f ms | %7.2f GFLOPS\n", 
           label, time_sec * 1000, gflops);
}

/* Memory bandwidth measurement */
double measure_bandwidth(size_t bytes, double time_sec) {
    return (bytes / 1e9) / time_sec;  /* GB/s */
}
