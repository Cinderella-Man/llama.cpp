#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Miner Llama (the Pascal fork) adds two ops to stock ggml: GGML_OP_GDN_CONV before GGML_OP_UNARY and GGML_OP_KQ_MASK
// after GGML_OP_GLU. Every op from GGML_OP_UNARY on therefore has another number than in stock llama.cpp, and a graph
// sent between a fork build and a stock build would run the wrong ops. The fork's major version (stock 8 + 100) makes
// each side refuse the other at the handshake: a client accepts only a server with its own major version.
#define RPC_PROTO_MAJOR_VERSION    108
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    2

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 104, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
