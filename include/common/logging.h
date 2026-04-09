#pragma once

#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

#ifndef PCCL_DEFAULT_LOG_LEVEL
#    define PCCL_DEFAULT_LOG_LEVEL spdlog::level::warn
#endif

namespace engine_c::common {

std::shared_ptr<spdlog::logger> getLogger();

#define PCCL_LOG(level, ...) SPDLOG_LOGGER_CALL(engine_c::common::getLogger(), level, __VA_ARGS__)
#define PCCL_LOG_TRACE(...) PCCL_LOG(spdlog::level::trace, __VA_ARGS__)
#define PCCL_LOG_DEBUG(...) PCCL_LOG(spdlog::level::debug, __VA_ARGS__)
#define PCCL_LOG_INFO(...) PCCL_LOG(spdlog::level::info, __VA_ARGS__)
#define PCCL_LOG_WARN(...) PCCL_LOG(spdlog::level::warn, __VA_ARGS__)
#define PCCL_LOG_ERROR(...) PCCL_LOG(spdlog::level::err, __VA_ARGS__)
#define PCCL_LOG_CRITICAL(...) PCCL_LOG(spdlog::level::critical, __VA_ARGS__)

#ifdef PCCL_DEBUG
#    define PCCL_DLOG(level, ...) PCCL_LOG(level, __VA_ARGS__)
#else
#    define PCCL_DLOG(level, ...) (void)0
#endif

#define PCCL_DLOG_TRACE(...) PCCL_DLOG(spdlog::level::trace, __VA_ARGS__)
#define PCCL_DLOG_DEBUG(...) PCCL_DLOG(spdlog::level::debug, __VA_ARGS__)
#define PCCL_DLOG_INFO(...) PCCL_DLOG(spdlog::level::info, __VA_ARGS__)
#define PCCL_DLOG_WARN(...) PCCL_DLOG(spdlog::level::warn, __VA_ARGS__)
#define PCCL_DLOG_ERROR(...) PCCL_DLOG(spdlog::level::err, __VA_ARGS__)
#define PCCL_DLOG_CRITICAL(...) PCCL_DLOG(spdlog::level::critical, __VA_ARGS__)

}  // namespace engine_c::common
