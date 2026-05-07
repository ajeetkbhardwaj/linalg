#include "../../include/linalg.h"
#include "../../include/config.h"
#include <stdlib.h>

workspace_t workspace_alloc(size_t size_bytes) {
    workspace_t ws;
    ws.buffer = (size_bytes > 0) ? malloc(size_bytes) : NULL;
    ws.size = ws.buffer ? size_bytes : 0;
    ws.is_active = (ws.buffer != NULL);
    return ws;
}

void workspace_free(workspace_t *ws) {
    if (ws && ws->buffer) {
        free(ws->buffer);
        ws->buffer = NULL;
        ws->size = 0;
        ws->is_active = false;
    }
}

size_t query_workspace_gemm(int m, int n, int k) {
    (void)m; (void)k;
    (void)n;
    return TILE_M * TILE_N * sizeof(float);
}

size_t query_workspace_getrf(int m, int n) {
    int min_mn = (m < n) ? m : n;
    return min_mn * sizeof(int);
}
