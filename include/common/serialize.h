#pragma once

#include <string>
#include <string_view>

namespace engine_c::common {

std::string serialize(const void* ptr, size_t nbyte);
void deserialize(void* ptr, std::string_view str);

}  // namespace engine_c::common
