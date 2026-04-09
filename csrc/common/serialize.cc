#include <common/serialize.h>
#include <mutex>

namespace engine_c::common {

std::string serialize(const void* ptr, size_t nbyte)
{
    static const char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(nbyte * 2 + 1);

    const unsigned char* byte_ptr = static_cast<const unsigned char*>(ptr);
    for (size_t i = 0; i < nbyte; ++i) {
        unsigned char byte = byte_ptr[i];
        result.push_back(hex_chars[byte >> 4]);
        result.push_back(hex_chars[byte & 0x0f]);
    }

    return result;
}

static unsigned char hex_char_to_value(char c)
{
    static unsigned char table[256] = {0};
    static std::once_flag table_flag;
    std::call_once(table_flag, [&]() {
        table[(int)'0'] = 0;
        table[(int)'1'] = 1;
        table[(int)'2'] = 2;
        table[(int)'3'] = 3;
        table[(int)'4'] = 4;
        table[(int)'5'] = 5;
        table[(int)'6'] = 6;
        table[(int)'7'] = 7;
        table[(int)'8'] = 8;
        table[(int)'9'] = 9;
        table[(int)'a'] = 10;
        table[(int)'b'] = 11;
        table[(int)'c'] = 12;
        table[(int)'d'] = 13;
        table[(int)'e'] = 14;
        table[(int)'f'] = 15;
    });
    return table[(int)c];
};

void deserialize(void* ptr, std::string_view str)
{
    unsigned char* byte_ptr = static_cast<unsigned char*>(ptr);
    size_t nbyte = str.length() / 2;

    for (size_t i = 0; i < nbyte; ++i) {
        char high_nibble = str[2 * i];
        char low_nibble = str[2 * i + 1];
        byte_ptr[i] = (hex_char_to_value(high_nibble) << 4) | hex_char_to_value(low_nibble);
    }
}

}  // namespace engine_c::common
