/* build.c - Project orchestrator for m4linalg */
#include "nobuild.h"
#include <string.h>
#include <libgen.h>

/* Global state for build */
static ProcGroup pg = {.max_parallel = 8};  /* Use M4's cores efficiently */
static char *obj_files[512];
static int obj_count = 0;

/* Callback: Compile a single .c file */
void compile_file_callback(const char *src_path, void *user_data) {
    (void)user_data;
    
    /* Generate object path: src/blas/level1.c -> obj/blas/level1.o */
    char obj_path[512];
    const char *src_rel = strstr(src_path, "src/");
    if (!src_rel) src_rel = src_path;
    
    /* Replace src/ with obj/ and .c with .o */
    snprintf(obj_path, sizeof(obj_path), "obj/%s", src_rel + 4);
    char *ext = strstr(obj_path, ".c");
    if (ext) strcpy(ext, ".o");
    
    /* Ensure output directory exists */
    char obj_dir[512];
    strncpy(obj_dir, obj_path, strlen(obj_path) - strlen(basename(obj_path)));
    obj_dir[strlen(obj_path) - strlen(basename(obj_path))] = '\0';
    ensure_dir(obj_dir);
    
    obj_files[obj_count++] = strdup(obj_path);
    
    /* Check if rebuild needed (incremental build) */
    if (needs_rebuild_with_deps(obj_path, src_path)) {
        printf("\033[32m[CC]\033[0m  %s\n", src_path);
        
        /* Compile with -MMD to generate dependency file */
        char *args[] = {
            COMPILER, CFLAGS, "-MMD", "-MP", "-c", 
            (char*)src_path, "-o", obj_path, NULL
        };
        run_async(&pg, args);
    }
}

/* Main build orchestration */
int main(int argc, char **argv) {
    const char *build_bin = argv[0];
    
    /* Self-rebuild if build.c changed */
    if (rebuild_self_if_needed(build_bin, "build.c")) {
        char *new_argv[argc + 1];
        memcpy(new_argv, argv, argc * sizeof(char*));
        new_argv[argc] = NULL;
        execv(build_bin, new_argv);
        perror("execv failed after rebuild");
        return 1;
    }
    
    /* Handle command line arguments */
    if (argc > 1) {
        if (strcmp(argv[1], "clean") == 0) {
            printf("\033[33mCleaning build directories...\033[0m\n");
            return system("rm -rf obj lib bin");
        }
        if (strcmp(argv[1], "install") == 0) {
            printf("\033[36mInstalling m4linalg to /usr/local/...\033[0m\n");
            if (system("mkdir -p /usr/local/include/m4linalg && cp -r include/* /usr/local/include/m4linalg/ && cp lib/libm4linalg.a lib/libm4linalg.dylib /usr/local/lib/") != 0) {
                fprintf(stderr, "\033[31mInstallation failed.\033[0m Try running with sudo: sudo ./build install\n");
                return 1;
            }
            printf("\033[32mInstallation complete!\033[0m\n");
            printf("You can now include <m4linalg/m4linalg.h> in your C files.\n");
            printf("Compile using: clang your_app.c -lm4linalg -framework Accelerate\n");
            return 0;
        }
    }
    
    printf("\033[1;36m Building m4linalg for Apple M4\033[0m\n");
    printf("   Compiler: %s\n", COMPILER);
    printf("   Flags: -mcpu=apple-m4 -ffast-math -MMD\n");
    printf("   Parallel jobs: %d\n\n", pg.max_parallel);
    
    /* Ensure output directories */
    ensure_dir("obj");
    ensure_dir("lib");
    
    /* Phase 1: Compile all source files recursively */
    printf("\033[1;34m[PHASE 1]\033[0m Compiling source modules...\n");
    walk_dir_recursive("src", compile_file_callback, NULL);
    sync_wait_all(&pg);
    
    /* Phase 2: Create static library */
    printf("\033[1;34m[PHASE 2]\033[0m Creating static library lib/libm4linalg.a\n");
    system("rm -f lib/libm4linalg.a");
    char *ar_cmd[520] = {"ar", "rcs", "lib/libm4linalg.a"};
    for (int i = 0; i < obj_count; i++) {
        ar_cmd[i + 3] = obj_files[i];
    }
    ar_cmd[obj_count + 3] = NULL;
    
    pid_t ar_pid = fork();
    if (ar_pid == 0) {
        execvp("ar", ar_cmd);
        perror("ar failed");
        exit(1);
    }
    waitpid(ar_pid, NULL, 0);
    
    /* Phase 3: Create dynamic library for Python bindings */
    printf("\033[1;34m[PHASE 3]\033[0m Creating dynamic library lib/libm4linalg.dylib\n");
    system("rm -f lib/libm4linalg.dylib");
    int dylib_status = system("clang -shared -O3 -mcpu=apple-m4 -ffast-math -o lib/libm4linalg.dylib obj/*/*.o -framework Accelerate");
    if (dylib_status != 0) {
        fprintf(stderr, "\033[31mWarning: Failed to create dynamic library.\033[0m\n");
    }
    
    /* Summary */
    printf("\n\033[1;35m✅ linalg build complete!\033[0m\n");
    printf("   Static Library: lib/libm4linalg.a (%d object files)\n", obj_count);
    if (dylib_status == 0) printf("   Dynamic Library: lib/libm4linalg.dylib\n");
    printf("\n\033[33mUsage:\033[0m\n");
    printf("   # In your project:\n");
    printf("   clang -I./include -L./lib -lm4linalg -framework Accelerate your_app.c\n\n");
    
    return 0;
}