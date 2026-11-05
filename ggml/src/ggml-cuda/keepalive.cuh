#pragma once

// GGML_CUDA_KEEPALIVE: a small block per card that holds the card's boost clock while a layer split decodes. The
// settings and the controller are in keepalive-policy.h; this is what the rest of the CUDA backend calls. Every call
// returns at once while the switch is off.

#include "keepalive-policy.h"

// devices are physical CUDA devices (ggml_cuda_get_physical_device), the cards the blocks run on

// before a graph's work is enqueued on the device; capture: a CUDA graph capture follows
void ggml_cuda_keepalive_graph_begin(int device, bool capture);

// after a graph's work is enqueued on the device
void ggml_cuda_keepalive_graph_end(int device);

// before anything that waits for every stream of a card or of every card (cudaFree, cudaFreeHost, cudaHostAlloc,
// cudaMallocHost, cudaHostRegister/Unregister, cuMemUnmap, cudaDeviceSynchronize): every block ended
void ggml_cuda_keepalive_quiesce();

// at a backend's free: the card's runs and the SM clock they measured, logged and reset
void ggml_cuda_keepalive_log(int device);

// llama-server took a request (GGML_CUDA_KEEPALIVE=request); the CUDA registry's "ggml_backend_cuda_keepalive_kick"
void ggml_backend_cuda_keepalive_kick(void);
