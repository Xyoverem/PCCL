#pragma once

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <unordered_map>
#include <string>
#include <mutex>
#include <vector>
#include <memory>

namespace engine_c::common {

class SocketInstance
{
   private:
    SocketInstance();
    ~SocketInstance();
    SocketInstance(const SocketInstance&) = delete;
    SocketInstance& operator=(const SocketInstance&) = delete;

    std::thread server_thread;
    int server_fd;
    int rank;
    bool running;

    std::unordered_map<std::string, int> entry_to_fd;
    std::mutex map_mutex;

    std::vector<std::unique_ptr<std::thread>> worker_threads;
    std::mutex worker_mutex;

    void handle_request(int client_socket);
    void server_loop();

   public:
    static SocketInstance& getInstance()
    {
        static SocketInstance instance;
        return instance;
    }

    static void start(int rank);
    static void add_fd(const std::string& entry, int fd);
    static void remove_fd(const std::string& entry);
    static int get_remote_fd(int rank, const std::string& entry);
    static void stop();
};

}  // namespace engine_c::common
