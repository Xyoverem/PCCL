#pragma once

#include <string>
#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <exception>

namespace engine_c {
namespace utils {

enum class ErrorSeverity {
  INFO,
  WARNING,
  ERROR,
  CRITICAL
};

enum class ErrorCode {
  SUCCESS = 0,

  NETWORK_ERROR = 1000,
  CONNECTION_FAILED = 1001,
  CONNECTION_LOST = 1002,
  TIMEOUT_ERROR = 1003,
  NETWORK_UNREACHABLE = 1004,

  MEMORY_ERROR = 2000,
  ALLOCATION_FAILED = 2001,
  OUT_OF_MEMORY = 2002,
  INVALID_POINTER = 2003,

  PROTOCOL_ERROR = 3000,
  INVALID_MESSAGE = 3001,
  PROTOCOL_MISMATCH = 3002,
  AUTHENTICATION_FAILED = 3003,

  RESOURCE_ERROR = 4000,
  RESOURCE_BUSY = 4001,
  RESOURCE_EXHAUSTED = 4002,
  DEVICE_UNAVAILABLE = 4003,

  CONFIG_ERROR = 5000,
  INVALID_CONFIG = 5001,
  MISSING_CONFIG = 5002,
  CONFIG_VERSION_MISMATCH = 5003,

  SYSTEM_ERROR = 6000,
  INTERNAL_ERROR = 6001,
  NOT_IMPLEMENTED = 6002,
  INVALID_STATE = 6003,

  TIMEOUT = 7000,
  CONNECTION_TIMEOUT = 7001,
  OPERATION_TIMEOUT = 7002,
  RETRY_EXHAUSTED = 7003
};

struct ErrorInfo {
  ErrorCode code;
  ErrorSeverity severity;
  std::string message;
  std::string component;
  std::string function;
  int line_number;
  std::chrono::system_clock::time_point timestamp;
  std::string stack_trace;

  ErrorInfo(ErrorCode c, ErrorSeverity s, const std::string& msg,
           const std::string& comp, const std::string& func = "", int line = 0)
    : code(c), severity(s), message(msg), component(comp), function(func),
      line_number(line), timestamp(std::chrono::system_clock::now()) {}
};

class Exception : public std::exception {
public:
  Exception(ErrorCode code, const std::string& message, const std::string& component = "");
  Exception(const ErrorInfo& error_info);

  const char* what() const noexcept override;
  ErrorCode getErrorCode() const { return error_info_.code; }
  ErrorSeverity getSeverity() const { return error_info_.severity; }
  const std::string& getComponent() const { return error_info_.component; }

private:
  ErrorInfo error_info_;
  mutable std::string what_message_;
};

class RecoveryStrategy {
public:
  virtual ~RecoveryStrategy() = default;
  virtual bool canHandle(const ErrorInfo& error) = 0;
  virtual bool recover(const ErrorInfo& error) = 0;
  virtual std::string getName() const = 0;
};

using RecoveryStrategyPtr = std::shared_ptr<RecoveryStrategy>;

class RetryStrategy : public RecoveryStrategy {
public:
  RetryStrategy(int max_attempts = 3, int base_delay_ms = 100, double backoff_factor = 2.0);

  bool canHandle(const ErrorInfo& error) override;
  bool recover(const ErrorInfo& error) override;
  std::string getName() const override { return "Retry"; }

private:
  int max_attempts_;
  int base_delay_ms_;
  double backoff_factor_;
};

class ReconnectionStrategy : public RecoveryStrategy {
public:
  ReconnectionStrategy(int max_attempts = 5, int timeout_ms = 5000);

  bool canHandle(const ErrorInfo& error) override;
  bool recover(const ErrorInfo& error) override;
  std::string getName() const override { return "Reconnection"; }

private:
  int max_attempts_;
  int timeout_ms_;
};

class FallbackStrategy : public RecoveryStrategy {
public:
  FallbackStrategy(const std::string& fallback_component);

  bool canHandle(const ErrorInfo& error) override;
  bool recover(const ErrorInfo& error) override;
  std::string getName() const override { return "Fallback"; }

private:
  std::string fallback_component_;
};

class ResourceCleanupStrategy : public RecoveryStrategy {
public:
  ResourceCleanupStrategy();

  bool canHandle(const ErrorInfo& error) override;
  bool recover(const ErrorInfo& error) override;
  std::string getName() const override { return "ResourceCleanup"; }
};

class ErrorHandler {
public:
  static ErrorHandler& getInstance();

  void reportError(const ErrorInfo& error);
  void reportError(ErrorCode code, ErrorSeverity severity, const std::string& message,
                  const std::string& component, const std::string& function = "", int line = 0);

  bool handleError(const ErrorInfo& error);
  void addRecoveryStrategy(RecoveryStrategyPtr strategy);
  void removeRecoveryStrategy(const std::string& name);

  void setErrorCallback(std::function<void(const ErrorInfo&)> callback);
  void setFatalErrorCallback(std::function<void(const ErrorInfo&)> callback);

  std::vector<ErrorInfo> getRecentErrors(size_t max_count = 100) const;
  void clearErrors();

  void enableLogging(bool enable) { logging_enabled_ = enable; }
  bool isLoggingEnabled() const { return logging_enabled_; }

  void setMaxStoredErrors(size_t max_count) { max_stored_errors_ = max_count; }

  ErrorSeverity getMinimumLogSeverity() const { return min_log_severity_; }
  void setMinimumLogSeverity(ErrorSeverity severity) { min_log_severity_ = severity; }

private:
  ErrorHandler() = default;
  ~ErrorHandler() = default;

  mutable std::mutex mutex_;
  std::vector<ErrorInfo> error_history_;
  std::vector<RecoveryStrategyPtr> recovery_strategies_;
  std::function<void(const ErrorInfo&)> error_callback_;
  std::function<void(const ErrorInfo&)> fatal_error_callback_;

  std::atomic<bool> logging_enabled_{true};
  std::atomic<size_t> max_stored_errors_{1000};
  std::atomic<ErrorSeverity> min_log_severity_{ErrorSeverity::WARNING};

  void logError(const ErrorInfo& error);
  std::string errorCodeToString(ErrorCode code) const;
  std::string severityToString(ErrorSeverity severity) const;
};

class ScopedErrorHandler {
public:
  ScopedErrorHandler(const std::string& component);
  ~ScopedErrorHandler();

  void reportError(ErrorCode code, ErrorSeverity severity, const std::string& message,
                  const std::string& function = "", int line = 0);

private:
  std::string component_;
};

class TimeoutHandler {
public:
  TimeoutHandler(int timeout_ms);

  bool start();
  bool stop();
  bool isExpired() const;
  void reset();

  int getRemainingMs() const;

private:
  int timeout_ms_;
  std::chrono::steady_clock::time_point start_time_;
  std::atomic<bool> running_{false};
};

class CircuitBreaker {
public:
  enum class State {
    CLOSED,
    OPEN,
    HALF_OPEN
  };

  CircuitBreaker(int failure_threshold = 5, int timeout_ms = 60000);

  bool call(std::function<bool()> operation);
  State getState() const { return state_; }
  int getFailureCount() const { return failure_count_; }

  void reset();

private:
  State state_;
  int failure_threshold_;
  int timeout_ms_;
  std::atomic<int> failure_count_{0};
  std::chrono::steady_clock::time_point last_failure_time_;
  mutable std::mutex mutex_;
};

#define PCCL_REPORT_ERROR(code, severity, message) \
  engine_c::utils::ErrorHandler::getInstance().reportError( \
    code, severity, message, __FILE__, __FUNCTION__, __LINE__)

#define PCCL_HANDLE_ERROR(code, severity, message) \
  do { \
    if (!engine_c::utils::ErrorHandler::getInstance().handleError( \
          engine_c::utils::ErrorInfo(code, severity, message, \
                                    __FILE__, __FUNCTION__, __LINE__))) { \
      throw engine_c::utils::Exception(code, message, __FILE__); \
    } \
  } while(0)

#define PCCL_SCOPED_HANDLER(name) \
  engine_c::utils::ScopedErrorHandler _scoped_handler(name)

#define PCCL_RETRY(operation, max_attempts) \
  engine_c::utils::RetryStrategy retry(max_attempts); \
  if (!retry.recover(engine_c::utils::ErrorInfo( \
        engine_c::utils::ErrorCode::RETRY_EXHAUSTED, \
        engine_c::utils::ErrorSeverity::ERROR, \
        "Operation failed after retries", __FILE__, __FUNCTION__, __LINE__))) { \
    PCCL_HANDLE_ERROR(engine_c::utils::ErrorCode::RETRY_EXHAUSTED, \
                     engine_c::utils::ErrorSeverity::ERROR, \
                     "Operation failed after maximum retries"); \
  }

}
}