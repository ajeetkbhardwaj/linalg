/* nobuild.h - Professional build engine for m4linalg */
#ifndef NOBUILD_H
#define NOBUILD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <libgen.h>

/* === Professional Configuration for Apple M4 === */
#define COMPILER "clang"
/* 
 * M4-specific optimizations:
 * - -O3: Maximum optimization
 * - -ffast-math: Enable aggressive FP optimizations (safe for well-conditioned LA)
 * - -mcpu=apple-m4: Target M4 instruction set including AMX/SME hints [[1]][[2]]
 * - -Iinclude: Header search path
 * - -fno-math-errno: Faster math by not setting errno
 */
#define CFLAGS "-O3", "-ffast-math", "-fno-math-errno", \
               "-Wall", "-Wextra", "-Wpedantic", \
               "-mcpu=apple-m4", "-Iinclude", "-DNDEBUG"

/* === Dependency Tracking: Parse .d files from clang -MMD === */
/* Format: target.o: src.c header1.h header2.h \ */
/*         header3.h ... (continued with backslash-newline) [[29]][[31]] */
bool needs_rebuild_with_deps(const char *obj_path, const char *src_path) {
    struct stat obj_st, src_st;
    
    /* No object file? Must build */
    if (stat(obj_path, &obj_st) < 0) return true;
    if (stat(src_path, &src_st) < 0) return false;
    
    /* Source newer than object? Rebuild */
    if (src_st.st_mtime > obj_st.st_mtime) return true;
    
    /* Check .d dependency file for header changes */
    char dep_path[512];
    snprintf(dep_path, sizeof(dep_path), "%.*s.d", 
             (int)(strlen(obj_path) - strlen(".o")), obj_path);
    
    FILE *f = fopen(dep_path, "r");
    if (!f) return false;  /* No .d file yet, assume clean */
    
    char line[4096];
    bool header_changed = false;
    
    while (fgets(line, sizeof(line), f)) {
        /* Skip target: prefix */
        char *colon = strchr(line, ':');
        if (!colon) continue;
        char *deps = colon + 1;
        
        /* Tokenize dependencies (space/backslash/newline separated) */
        char *token = strtok(deps, " \\\n\r\t");
        while (token) {
            /* Only check .h files that actually exist */
            if (strstr(token, ".h") && strlen(token) > 2) {
                struct stat h_st;
                if (stat(token, &h_st) == 0 && h_st.st_mtime > obj_st.st_mtime) {
                    header_changed = true;
                    break;
                }
            }
            token = strtok(NULL, " \\\n\r\t");
        }
        if (header_changed) break;
    }
    fclose(f);
    return header_changed;
}

/* === Recursive Directory Scanner === */
typedef void (*FileCallback)(const char *path, void *user_data);

void walk_dir_recursive(const char *dir_path, FileCallback cb, void *user_data) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "Warning: Cannot open directory %s: %s\n", 
                dir_path, strerror(errno));
        return;
    }
    
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        /* Skip hidden files and . / .. */
        if (ent->d_name[0] == '.') continue;
        
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir_path, ent->d_name);
        
        struct stat st;
        if (stat(path, &st) < 0) continue;
        
        if (S_ISDIR(st.st_mode)) {
            /* Recurse into subdirectory */
            walk_dir_recursive(path, cb, user_data);
        } else if (strstr(ent->d_name, ".c") && !strstr(ent->d_name, ".o")) {
            /* Found a .c source file */
            cb(path, user_data);
        }
    }
    closedir(dir);
}

/* === Parallel Execution with Process Groups === */
typedef struct {
    pid_t pids[128];
    size_t count;
    int max_parallel;
} ProcGroup;

void run_async(ProcGroup *pg, char **args) {
    if (pg->count >= pg->max_parallel) {
        /* Wait for one to finish before spawning more */
        waitpid(pg->pids[0], NULL, 0);
        memmove(pg->pids, pg->pids + 1, sizeof(pid_t) * (--pg->count));
    }
    
    pid_t pid = fork();
    if (pid == 0) {
        /* Child process */
        execvp(args[0], args);
        perror("execvp failed");
        exit(127);
    } else if (pid > 0) {
        pg->pids[pg->count++] = pid;
    }
}

void sync_wait_all(ProcGroup *pg) {
    for (size_t i = 0; i < pg->count; i++) {
        int status;
        waitpid(pg->pids[i], &status, 0);
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Build step failed with exit code %d\n", 
                    WEXITSTATUS(status));
        }
    }
    pg->count = 0;
}

/* === Self-Rebuilding Build Script === */
bool rebuild_self_if_needed(const char *build_bin, const char *build_src) {
    struct stat st_src, st_bin;
    if (stat(build_src, &st_src) < 0) return false;
    if (stat(build_bin, &st_bin) < 0) return true;  /* Binary doesn't exist */
    
    if (st_src.st_mtime > st_bin.st_mtime) {
        printf("\033[33m[REBUILD] build.c updated. Recompiling build engine...\033[0m\n");
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "%s -O3 %s -o %s.tmp && mv %s.tmp %s",
                 COMPILER, build_src, build_bin, build_bin, build_bin);
        if (system(cmd) != 0) {
            fprintf(stderr, "Failed to rebuild build.c\n");
            return false;
        }
        printf("\033[32m[REBUILD] Build engine updated. Restarting...\033[0m\n");
        return true;  /* Signal caller to re-exec */
    }
    return false;
}

/* === Utility: Ensure directory exists === */
bool ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) return true;
    return mkdir(path, 0755) == 0 || errno == EEXIST;
}

#endif /* NOBUILD_H */