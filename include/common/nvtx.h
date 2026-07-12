#pragma once

#include <nvtx3/nvToolsExt.h>

namespace engine_c {

class NvtxRange {
   public:
    explicit NvtxRange(const char* message) {
        nvtxRangePushA(message);
    }

    ~NvtxRange() {
        nvtxRangePop();
    }
};

}  // namespace engine_c
