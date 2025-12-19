#include <utils/error_handling.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <execinfo.h>
#include <cxxabi.h>

namespace engine_c {
namespace utils {

Exception::Exception(ErrorCode code, const std::string& message, const std::string& component)
  : error_info_(code, ErrorSeverity::ERROR, message, component) {
}

Exception::Exception(const ErrorInfo& error_info)
  : error_info_(error_info) {
}

const char* Exception::what() const noexcept {
  what_message_ = "[" + std::to_string(static_cast<int>(error_info_.code)) + "] " +
                   error_info_.message + " (Component: " + error_info_.component + ")";
  return what_message_.c_str();
}

RetryStrategy::RetryStrategy(int max_attempts, int base_delay_ms, double backoff_factor)
  : max_attempts_(max_attempts), base_delay_ms_(base_delay_ms), backoff_factor_(backoff_factor) {
}

bool RetryStrategy::canHandle(const ErrorInfo& error) {
  return error.code == ErrorCode::NETWORK_ERROR ||
         error.code == ErrorCode::CONNECTION_FAILED ||
         error.code == ErrorCode::TIMEOUT_ERROR ||
         error.code == ErrorCode::CONNECTION_TIMEOUT ||
         error.code == ErrorCode::OPERATION_TIMEOUT ||
         error.code == ErrorCode::RETRY_EXHAUSTED;
}

bool RetryStrategy::recover(const ErrorInfo& error) {
  if (!canHandle(error)) {
    return false;
  }

  for (int attempt = 0; attempt < max_attempts_; ++attempt) {
    if (attempt > 0) {
      int delay = static_cast<int>(base_delay_ms_ * std::pow(backoff_factor_, attempt - 1));
      std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }

    return true;
  }

  return false;
}

ReconnectionStrategy::ReconnectionStrategy(int max_attempts, int timeout_ms)
  : max_attempts_(max_attempts), timeout_ms_(timeout_ms) {
}

bool ReconnectionStrategy::canHandle(const ErrorInfo& error) {
  return error.code == ErrorCode::CONNECTION_LOST ||
         error.code == ErrorCode::CONNECTION_FAILED ||
         error.code == ErrorCode::NETWORK_UNREACHABLE;
}

bool ReconnectionStrategy::recover(const ErrorInfo& error) {
  if (!canHandle(error)) {
    return false;
  }

  for (int attempt = 0; attempt < max_attempts_; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    return true;
  }

  return false;
}

FallbackStrategy::FallbackStrategy(const std::string& fallback_component)
  : fallback_component_(fallback_component) {
}

bool FallbackStrategy::canHandle(const ErrorInfo& error) {
  return error.severity == ErrorSeverity::CRITICAL ||
         error.code == ErrorCode::DEVICE_UNAVAILABLE ||
         error.code == ErrorCode::RESOURCE_EXHAUSTED;
}

bool FallbackStrategy::recover(const ErrorInfo& error) {
  if (!canHandle(error)) {
    return false;
  }

  return !fallback_component_.empty();
}

ResourceCleanupStrategy::ResourceCleanupStrategy() {
}

bool ResourceCleanupStrategy::canHandle(const ErrorInfo& error) {
  return error.code == ErrorCode::OUT_OF_MEMORY ||
         error.code == ErrorCode::ALLOCATION_FAILED ||
         error.code == ErrorCode::RESOURCE_EXHAUSTED ||
         error.code == ErrorCode::RESOURCE_BUSY;
}

bool ResourceCleanupStrategy::recover(const ErrorInfo& error) {
  if (!canHandle(error)) {
    return false;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  return true;
}

ErrorHandler& ErrorHandler::getInstance() {
  static ErrorHandler instance;
  return instance;
}

void ErrorHandler::reportError(const ErrorInfo& error) {
  std::lock_guard<std::mutex> lock(mutex_);

  error_history_.push_back(error);

  if (error_history_.size() > max_stored_errors_) {
    error_history_.erase(error_history_.begin());
  }

  if (logging_enabled_ && error.severity >= min_log_severity_) {
    logError(error);
  }

  if (error_callback_) {
    error_callback_(error);
  }

  if (error.severity == ErrorSeverity::CRITICAL && fatal_error_callback_) {
    fatal_error_callback_(error);
  }
}

void ErrorHandler::reportError(ErrorCode code, ErrorSeverity severity, const std::string& message,
                               const std::string& component, const std::string& function, int line) {
  ErrorInfo error(code, severity, message, component, function, line);
  reportError(error);
}

bool ErrorHandler::handleError(const ErrorInfo& error) {
  bool handled = false;

  for (auto& strategy : recovery_strategies_) {
    if (strategy->canHandle(error)) {
      try {
        if (strategy->recover(error)) {
          handled = true;
          break;
        }
      } catch (const std::exception& e) {
        reportError(ErrorCode::INTERNAL_ERROR, ErrorSeverity::ERROR,
                   "Recovery strategy failed: " + std::string(e.what()),
                   "ErrorHandler", __FUNCTION__, __LINE__);
      }
    }
  }

  return handled;
}

void ErrorHandler::addRecoveryStrategy(RecoveryStrategyPtr strategy) {
  std::lock_guard<std::mutex> lock(mutex_);
  recovery_strategies_.push_back(strategy);
}

void ErrorHandler::removeRecoveryStrategy(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  recovery_strategies_.erase(
    std::remove_if(recovery_strategies_.begin(), recovery_strategies_.end(),
                   [&](const RecoveryStrategyPtr& strategy) {
                     return strategy->getName() == name;
                   }),
    recovery_strategies_.end());
}

void ErrorHandler::setErrorCallback(std::function<void(const ErrorInfo&)> callback) {
  error_callback_ = callback;
}

void ErrorHandler::setFatalErrorCallback(std::function<void(const ErrorInfo&)> callback) {
  fatal_error_callback_ = callback;
}

std::vector<ErrorInfo> ErrorHandler::getRecentErrors(size_t max_count) const {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t start = (error_history_.size() > max_count) ?
                 (error_history_.size() - max_count) : 0;

  return std::vector<ErrorInfo>(error_history_.begin() + start, error_history_.end());
}

void ErrorHandler::clearErrors() {
  std::lock_guard<std::mutex> lock(mutex_);
  error_history_.clear();
}

void ErrorHandler::logError(const ErrorInfo& error) {
  std::string severity_str = severityToString(error.severity);
  std::string code_str = errorCodeToString(error.code);

  auto time_t = std::chrono::system_clock::to_time_t(error.timestamp);
  auto tm = *std::localtime(&time_t);

  std::cout << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] "
            << "[" << severity_str << "] "
            << "[" << code_str << "] "
            << "[" << error.component << "] "
            << error.message;

  if (!error.function.empty()) {
    std::cout << " (" << error.function << ":" << error.line_number << ")";
  }

  std::cout << std::endl;

  if (error.severity == ErrorSeverity::ERROR || error.severity == ErrorSeverity::CRITICAL) {
    std::ofstream log_file("pccl_errors.log", std::ios::app);
    if (log_file.is_open()) {
      log_file << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] "
               << "[" << severity_str << "] "
               << "[" << code_str << "] "
               << "[" << error.component << "] "
               << error.message;

      if (!error.function.empty()) {
        log_file << " (" << error.function << ":" << error.line_number << ")";
      }

      log_file << std::endl;
    }
  }
}

std::string ErrorHandler::errorCodeToString(ErrorCode code) const {
  switch (code) {
    case ErrorCode::SUCCESS: return "SUCCESS";
    case ErrorCode::NETWORK_ERROR: return "NETWORK_ERROR";
    case ErrorCode::CONNECTION_FAILED: return "CONNECTION_FAILED";
    case ErrorCode::CONNECTION_LOST: return "CONNECTION_LOST";
    case ErrorCode::TIMEOUT_ERROR: return "TIMEOUT_ERROR";
    case ErrorCode::NETWORK_UNREACHABLE: return "NETWORK_UNREACHABLE";
    case ErrorCode::MEMORY_ERROR: return "MEMORY_ERROR";
    case ErrorCode::ALLOCATION_FAILED: return "ALLOCATION_FAILED";
    case ErrorCode::OUT_OF_MEMORY: return "OUT_OF_MEMORY";
    case ErrorCode::INVALID_POINTER: return "INVALID_POINTER";
    case ErrorCode::PROTOCOL_ERROR: return "PROTOCOL_ERROR";
    case ErrorCode::INVALID_MESSAGE: return "INVALID_MESSAGE";
    case ErrorCode::PROTOCOL_MISMATCH: return "PROTOCOL_MISMATCH";
    case ErrorCode::AUTHENTICATION_FAILED: return "AUTHENTICATION_FAILED";
    case ErrorCode::RESOURCE_ERROR: return "RESOURCE_ERROR";
    case ErrorCode::RESOURCE_BUSY: return "RESOURCE_BUSY";
    case ErrorCode::RESOURCE_EXHAUSTED: return "RESOURCE_EXHAUSTED";
    case ErrorCode::DEVICE_UNAVAILABLE: return "DEVICE_UNAVAILABLE";
    case ErrorCode::CONFIG_ERROR: return "CONFIG_ERROR";
    case ErrorCode::INVALID_CONFIG: return "INVALID_CONFIG";
    case ErrorCode::MISSING_CONFIG: return "MISSING_CONFIG";
    case ErrorCode::CONFIG_VERSION_MISMATCH: return "CONFIG_VERSION_MISMATCH";
    case ErrorCode::SYSTEM_ERROR: return "SYSTEM_ERROR";
    case ErrorCode::INTERNAL_ERROR: return "INTERNAL_ERROR";
    case ErrorCode::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
    case ErrorCode::INVALID_STATE: return "INVALID_STATE";
    case ErrorCode::TIMEOUT: return "TIMEOUT";
    case ErrorCode::CONNECTION_TIMEOUT: return "CONNECTION_TIMEOUT";
    case ErrorCode::OPERATION_TIMEOUT: return "OPERATION_TIMEOUT";
    case ErrorCode::RETRY_EXHAUSTED: return "RETRY_EXHAUSTED";
    default: return "UNKNOWN";
  }
}

std::string ErrorHandler::severityToString(ErrorSeverity severity) const {
  switch (severity) {
    case ErrorSeverity::INFO: return "INFO";
    case ErrorSeverity::WARNING: return "WARN";
    case ErrorSeverity::ERROR: return "ERROR";
    case ErrorSeverity::CRITICAL: return "CRIT";
    default: return "UNKNOWN";
  }
}

ScopedErrorHandler::ScopedErrorHandler(const std::string& component)
  : component_(component) {
}

void ScopedErrorHandler::reportError(ErrorCode code, ErrorSeverity severity, const std::string& message,
                                    const std::string& function, int line) {
  ErrorHandler::getInstance().reportError(code, severity, message, component_, function, line);
}

TimeoutHandler::TimeoutHandler(int timeout_ms)
  : timeout_ms_(timeout_ms) {
}

bool TimeoutHandler::start() {
  start_time_ = std::chrono::steady_clock::now();
  running_ = true;
  return true;
}

bool TimeoutHandler::stop() {
  running_ = false;
  return true;
}

bool TimeoutHandler::isExpired() const {
  if (!running_) {
    return false;
  }

  auto elapsed = std::chrono::steady_clock::now() - start_time_;
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  return elapsed_ms >= timeout_ms_;
}

int TimeoutHandler::getRemainingMs() const {
  if (!running_) {
    return timeout_ms_;
  }

  auto elapsed = std::chrono::steady_clock::now() - start_time_;
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  return std::max(0, timeout_ms_ - static_cast<int>(elapsed_ms));
}

void TimeoutHandler::reset() {
  start_time_ = std::chrono::steady_clock::now();
  running_ = true;
}

CircuitBreaker::CircuitBreaker(int failure_threshold, int timeout_ms)
  : state_(State::CLOSED), failure_threshold_(failure_threshold), timeout_ms_(timeout_ms) {
}

bool CircuitBreaker::call(std::function<bool()> operation) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (state_ == State::OPEN) {
    auto time_since_failure = std::chrono::steady_clock::now() - last_failure_time_;
    auto timeout_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_failure).count();

    if (timeout_ms >= timeout_ms_) {
      state_ = State::HALF_OPEN;
    } else {
      return false;
    }
  }

  try {
    bool result = operation();

    if (result) {
      failure_count_ = 0;
      state_ = State::CLOSED;
    } else {
      failure_count_++;
      last_failure_time_ = std::chrono::steady_clock::now();

      if (failure_count_ >= failure_threshold_) {
        state_ = State::OPEN;
      }
    }

    return result;

  } catch (...) {
    failure_count_++;
    last_failure_time_ = std::chrono::steady_clock::now();

    if (failure_count_ >= failure_threshold_) {
      state_ = State::OPEN;
    }

    return false;
  }
}

void CircuitBreaker::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  state_ = State::CLOSED;
  failure_count_ = 0;
}

}
}