#include <spdlog/sinks/stdout_color_sinks.h>
#include <common/logging.h>
#include <common/environ.h>
#include <fmt/format.h>

static auto select_spdlog_level(std::string_view level_str)
{
    if (level_str == "TRACE") return spdlog::level::trace;
    if (level_str == "DEBUG") return spdlog::level::debug;
    if (level_str == "INFO") return spdlog::level::info;
    if (level_str == "WARN") return spdlog::level::warn;
    if (level_str == "ERROR") return spdlog::level::err;
    if (level_str == "CRITICAL") return spdlog::level::critical;
    return PCCL_DEFAULT_LOG_LEVEL;
}

std::shared_ptr<spdlog::logger> engine_c::common::getLogger()
{
    static auto logger = []() {
        auto pccl_log_level = engine_c::common::Environs::getEnvOrDefault("PCCL_LOG_LEVEL", "");
        auto log_level =
            pccl_log_level == "" ? PCCL_DEFAULT_LOG_LEVEL : select_spdlog_level(pccl_log_level);

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(log_level);

        auto logger = std::make_shared<spdlog::logger>("pccl", console_sink);
        logger->set_level(log_level);
        logger->flush_on(log_level);

        return logger;
    }();

    return logger;
}
