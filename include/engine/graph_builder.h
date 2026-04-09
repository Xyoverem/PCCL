#pragma once

#include <string>
#include <vector>
#include <engine/primitive.h>
#include <engine/ring_buffer.h>
#include <engine/fused_step.h>
#include <engine/workspace.h>
#include <plugins/registry.h>

namespace engine_c {

class GraphBuilder
{
   public:
    GraphBuilder() = default;
    ~GraphBuilder() = default;

    bool loadFromFile(const std::string& filename);
    bool build(DeviceWorkspace* workspace);
    bool buildFusedDescriptor(FusedStepDescriptor* desc);
    const std::string& collectiveType() const { return collective_type_; }

   private:
    std::vector<ProxyTrigger> operators_;
    std::vector<PrimitiveMeta> metadata_;
    int num_primitives_ = 0;
    std::string collective_type_;
};

}  // namespace engine_c
