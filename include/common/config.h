#pragma once

#include <cstddef>

namespace pccl::config {

// Buffer allocation
static constexpr long DEFAULT_BUFFER_SIZE = 1L << 30;       // 1GB
static constexpr long MIN_BUFFER_SIZE     = 64L * 1024 * 1024;  // 64MB
static constexpr long SIGNAL_SIZE         = 4096 * sizeof(int); // 16KB

// Memory layout regions
static constexpr long WORKSPACE_REGION_SIZE    = 256L * 1024 * 1024;
static constexpr long DEV_WORKSPACE_SIZE       = 4096;
static constexpr long PEER_SLOT_SIZE           = 4096;
static constexpr long GRAPH_META_SIZE          = 32L * 1024 * 1024;
static constexpr long PRIMITIVES_SIZE          = 30L * 1024 * 1024;
static constexpr long WORKING_META_SIZE        = 2L * 1024 * 1024;
static constexpr long QUEUE_DATA_SIZE          = 4L * 1024 * 1024;
static constexpr long QUEUE_META_SIZE          = 1024;
static constexpr long QUEUE_TOTAL_SIZE         = QUEUE_DATA_SIZE + QUEUE_META_SIZE;
static constexpr long CHANNEL_QUEUE_DATA_SIZE  = 4096;
static constexpr long CHANNEL_QUEUE_META_SIZE  = 64;
static constexpr long CHANNEL_QUEUE_TOTAL_SIZE = CHANNEL_QUEUE_DATA_SIZE + CHANNEL_QUEUE_META_SIZE;

// Execution limits
static constexpr int MAX_CHANNELS         = 8;
static constexpr int MAX_FUSED_STEPS      = 128;
static constexpr int MAX_FUSED_CHANNELS   = 8;
static constexpr int MAX_CHUNK_SIZE_BYTES = 4 * 1024 * 1024;  // 4MB
static constexpr int MIN_CHUNK_ELEMS      = 16384;

// Queue capacities
static constexpr int PROXY_QUEUE_CAPACITY = 256;
static constexpr int CE_PROXY_QUEUE_CAPACITY = 256;

}  // namespace pccl::config
