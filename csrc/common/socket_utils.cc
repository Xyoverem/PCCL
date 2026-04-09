#include <common.h>
#include <fmt/format.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <cstring>
#include <vector>
#include <algorithm>

namespace engine_c::common {

static constexpr char PROTOCOL_MAGIC[] = "PCCL_FD_SHARE";
static constexpr size_t MAX_STRING_LEN = 128;

static std::string generate_socket_id(int rank)
{
    return fmt::format("/tmp/pccl_{}.sock", rank);
}

static int send_fd(int socket, int fd)
{
    struct msghdr msg = {0};
    struct iovec iov[1];
    char buffer[1] = {'F'};

    char control_buffer[CMSG_SPACE(sizeof(int))];
    msg.msg_control = control_buffer;
    msg.msg_controllen = sizeof(control_buffer);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg)
        return -1;

    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));

    iov[0].iov_base = buffer;
    iov[0].iov_len = sizeof(buffer);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    return sendmsg(socket, &msg, 0) < 0 ? -1 : 0;
}

static int recv_fd(int socket)
{
    struct msghdr msg = {0};
    struct iovec iov[1];
    char buffer[1];

    char control_buffer[CMSG_SPACE(sizeof(int))];
    msg.msg_control = control_buffer;
    msg.msg_controllen = sizeof(control_buffer);

    iov[0].iov_base = buffer;
    iov[0].iov_len = sizeof(buffer);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    if (recvmsg(socket, &msg, 0) < 0)
        return -1;

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
        return -1;
    }

    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
    return fd;
}

static bool send_string(int socket, const std::string& str)
{
    uint32_t len = static_cast<uint32_t>(str.length());
    if (send(socket, &len, sizeof(len), 0) != sizeof(len))
        return false;
    if (len > 0 && send(socket, str.c_str(), len, 0) != static_cast<ssize_t>(len))
        return false;
    return true;
}

static bool recv_string(int socket, std::string& str)
{
    uint32_t len = 0;
    if (recv(socket, &len, sizeof(len), 0) != sizeof(len))
        return false;

    if (len > MAX_STRING_LEN)
        return false;

    if (len > 0) {
        std::vector<char> buffer(len + 1);
        if (recv(socket, buffer.data(), len, 0) != static_cast<ssize_t>(len))
            return false;
        buffer[len] = '\0';
        str.assign(buffer.data());
    } else {
        str.clear();
    }

    return true;
}

SocketInstance::SocketInstance() : server_fd(-1), running(false) {}

SocketInstance::~SocketInstance()
{
    stop();
}

void SocketInstance::start(int rank)
{
    auto& instance = getInstance();
    if (instance.running)
        return;

    instance.running = true;

    instance.server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (instance.server_fd < 0) {
        perror("socket");
        instance.running = false;
        return;
    }

    int flags = fcntl(instance.server_fd, F_GETFL, 0);
    if (flags < 0 || fcntl(instance.server_fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("fcntl");
        close(instance.server_fd);
        instance.running = false;
        return;
    }

    std::string socket_path = generate_socket_id(rank);
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    unlink(addr.sun_path);

    if (bind(instance.server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0 ||
        listen(instance.server_fd, 10) < 0) {
        perror("bind/listen");
        close(instance.server_fd);
        instance.running = false;
        return;
    }

    instance.server_thread = std::thread(&SocketInstance::server_loop, &instance);
}

void SocketInstance::server_loop()
{
    while (running) {
        struct pollfd pfd = {server_fd, POLLIN, 0};
        int ret = poll(&pfd, 1, 100);

        if (ret < 0) {
            if (errno != EINTR)
                break;
            continue;
        }

        if (ret > 0 && (pfd.revents & POLLIN)) {
            int client_fd = accept(server_fd, nullptr, nullptr);
            if (client_fd >= 0) {
                auto thread = std::make_unique<std::thread>(
                    [this, client_fd]() { handle_request(client_fd); });

                std::lock_guard<std::mutex> lock(worker_mutex);
                worker_threads.push_back(std::move(thread));
            }
        }

        std::lock_guard<std::mutex> lock(worker_mutex);
        worker_threads.erase(std::remove_if(worker_threads.begin(), worker_threads.end(),
                                            [](const std::unique_ptr<std::thread>& t) {
                                                return t && t->joinable() == false;
                                            }),
                             worker_threads.end());
    }
}

void SocketInstance::handle_request(int client_socket)
{
    struct timeval tv = {2, 0};
    setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::string magic, entry;

    if (!recv_string(client_socket, magic) || magic != PROTOCOL_MAGIC ||
        !recv_string(client_socket, entry) || entry.empty()) {
        close(client_socket);
        return;
    }

    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(map_mutex);
        auto it = entry_to_fd.find(entry);
        if (it != entry_to_fd.end())
            fd = it->second;
    }

    send_fd(client_socket, fd);
    close(client_socket);
}

void SocketInstance::add_fd(const std::string& entry, int fd)
{
    auto& instance = getInstance();
    if (entry.empty() || fd < 0)
        return;

    std::lock_guard<std::mutex> lock(instance.map_mutex);
    instance.entry_to_fd[entry] = fd;
}

void SocketInstance::remove_fd(const std::string& entry)
{
    auto& instance = getInstance();
    if (entry.empty())
        return;

    std::lock_guard<std::mutex> lock(instance.map_mutex);
    instance.entry_to_fd.erase(entry);
}

int SocketInstance::get_remote_fd(int rank, const std::string& entry)
{
    if (rank < 0 || entry.empty())
        return -1;

    std::string socket_path = generate_socket_id(rank);
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0)
        return -1;

    struct timeval tv = {2, 0};
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0 ||
        !send_string(sock, PROTOCOL_MAGIC) || !send_string(sock, entry)) {
        close(sock);
        return -1;
    }

    int fd = recv_fd(sock);
    close(sock);
    return fd;
}

void SocketInstance::stop()
{
    auto& instance = getInstance();
    if (!instance.running)
        return;

    instance.running = false;

    if (instance.server_fd >= 0) {
        close(instance.server_fd);
        instance.server_fd = -1;
    }

    if (instance.server_thread.joinable()) {
        instance.server_thread.join();
    }

    {
        std::lock_guard<std::mutex> lock(instance.worker_mutex);
        for (auto& thread : instance.worker_threads) {
            if (thread && thread->joinable()) {
                thread->join();
            }
        }
        instance.worker_threads.clear();
    }

    std::string socket_path = generate_socket_id(0);
    unlink(socket_path.c_str());

    std::lock_guard<std::mutex> lock(instance.map_mutex);
    instance.entry_to_fd.clear();
}

}  // namespace engine_c::common
