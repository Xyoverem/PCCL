#include <Python.h>
#include <network/network.h>
#include <communication/communicator.h>
#include <cluster/process_group.h>
#include <algorithms/communication_wrapper.h>
#include <memory>

namespace {

struct PyNetworkConnection {
  PyObject_HEAD
  engine_c::network::ConnectionPtr conn;
};

struct PyCommunicator {
  PyObject_HEAD
  std::unique_ptr<engine_c::communication::Communicator> comm;
};

struct PyProcessGroup {
  PyObject_HEAD
  std::shared_ptr<engine_c::ProcessGroup> group;
};

static PyObject* PyNetworkConnection_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PyNetworkConnection* self = (PyNetworkConnection*)type->tp_alloc(type, 0);
  if (self) {
    new (&self->conn) engine_c::network::ConnectionPtr();
  }
  return (PyObject*)self;
}

static void PyNetworkConnection_dealloc(PyNetworkConnection* self) {
  self->conn.~ConnectionPtr();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyNetworkConnection_connect(PyNetworkConnection* self, PyObject* args) {
  const char* ip;
  int port;

  if (!PyArg_ParseTuple(args, "si", &ip, &port)) {
    return NULL;
  }

  engine_c::network::NetworkAddress addr(ip, port);
  bool result = self->conn->connect(addr);

  return PyBool_FromLong(result);
}

static PyObject* PyNetworkConnection_send_message(PyNetworkConnection* self, PyObject* args) {
  PyBytesObject* data;
  int dest_rank, tag;

  if (!PyArg_ParseTuple(args, "Oii", &data, &dest_rank, &tag)) {
    return NULL;
  }

  char* buffer;
  Py_ssize_t size;
  PyBytes_AsStringAndSize((PyObject*)data, &buffer, &size);

  engine_c::network::MessageHeader header;
  header.message_id = tag;
  header.data_size = size;
  header.source_rank = 0;
  header.dest_rank = dest_rank;
  header.tag = tag;
  header.flags = 0;
  header.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();

  bool result = self->conn->sendMessage(header, buffer);

  return PyBool_FromLong(result);
}

static PyObject* PyNetworkConnection_recv_message(PyNetworkConnection* self, PyObject* args) {
  int source_rank, tag;
  Py_ssize_t max_size = 4096;

  if (!PyArg_ParseTuple(args, "ii|n", &source_rank, &tag, &max_size)) {
    return NULL;
  }

  engine_c::network::MessageHeader header;
  std::vector<char> buffer(max_size);

  bool result = self->conn->receiveMessage(header, buffer.data(), max_size);

  if (result) {
    return PyBytes_FromStringAndSize(buffer.data(), header.data_size);
  }

  Py_RETURN_NONE;
}

static PyMethodDef PyNetworkConnection_methods[] = {
  {"connect", (PyCFunction)PyNetworkConnection_connect, METH_VARARGS,
   "Connect to a remote endpoint"},
  {"send_message", (PyCFunction)PyNetworkConnection_send_message, METH_VARARGS,
   "Send a message to remote rank"},
  {"recv_message", (PyCFunction)PyNetworkConnection_recv_message, METH_VARARGS,
   "Receive a message from remote rank"},
  {NULL}
};

static PyTypeObject PyNetworkConnectionType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "pccl.network.NetworkConnection",
  sizeof(PyNetworkConnection),
  0,
  (destructor)PyNetworkConnection_dealloc,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  Py_TPFLAGS_DEFAULT,
  "PCCL Network Connection",
  0, 0, 0, 0, 0, 0,
  PyNetworkConnection_methods,
  0, 0, 0, 0, 0, 0, 0,
  PyNetworkConnection_new,
};

static PyObject* PyCommunicator_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PyCommunicator* self = (PyCommunicator*)type->tp_alloc(type, 0);
  if (self) {
    new (&self->comm) std::unique_ptr<engine_c::communication::Communicator>();
  }
  return (PyObject*)self;
}

static void PyCommunicator_dealloc(PyCommunicator* self) {
  self->comm.~unique_ptr();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyCommunicator_initialize(PyCommunicator* self, PyObject* args) {
  int rank, world_size;
  const char* network_type = "tcp";

  if (!PyArg_ParseTuple(args, "ii|s", &rank, &world_size, &network_type)) {
    return NULL;
  }

  engine_c::network::NetworkType type = engine_c::network::NetworkType::TCP_SOCKET;
  if (std::string(network_type) == "rdma") {
    type = engine_c::network::NetworkType::RDMA_VERBS;
  }

  self->comm = engine_c::communication::createCommunicator(type);
  bool result = self->comm->initialize(rank, world_size);

  return PyBool_FromLong(result);
}

static PyObject* PyCommunicator_allreduce(PyCommunicator* self, PyObject* args) {
  PyBytesObject* input_data;
  Py_ssize_t data_size;
  const char* reduce_op = "sum";

  if (!PyArg_ParseTuple(args, "On|s", &input_data, &data_size, &reduce_op)) {
    return NULL;
  }

  char* input_buffer;
  PyBytes_AsStringAndSize((PyObject*)input_data, &input_buffer, &data_size);

  std::vector<char> output_buffer(data_size);

  engine_c::communication::ReductionOp op = engine_c::communication::ReductionOp::SUM;
  if (std::string(reduce_op) == "max") {
    op = engine_c::communication::ReductionOp::MAX;
  } else if (std::string(reduce_op) == "min") {
    op = engine_c::communication::ReductionOp::MIN;
  }

  bool result = self->comm->allReduce(input_buffer, output_buffer.data(), data_size, op);

  if (result) {
    return PyBytes_FromStringAndSize(output_buffer.data(), data_size);
  }

  Py_RETURN_NONE;
}

static PyObject* PyCommunicator_broadcast(PyCommunicator* self, PyObject* args) {
  PyBytesObject* buffer_obj;
  int root_rank;

  if (!PyArg_ParseTuple(args, "Oi", &buffer_obj, &root_rank)) {
    return NULL;
  }

  char* buffer;
  Py_ssize_t size;
  PyBytes_AsStringAndSize((PyObject*)buffer_obj, &buffer, &size);

  bool result = self->comm->broadcast(buffer, size, root_rank);

  if (result) {
    Py_INCREF(buffer_obj);
    return (PyObject*)buffer_obj;
  }

  Py_RETURN_NONE;
}

static PyObject* PyCommunicator_send(PyCommunicator* self, PyObject* args) {
  PyBytesObject* data;
  int dest_rank, tag;

  if (!PyArg_ParseTuple(args, "Oii", &data, &dest_rank, &tag)) {
    return NULL;
  }

  char* buffer;
  Py_ssize_t size;
  PyBytes_AsStringAndSize((PyObject*)data, &buffer, &size);

  bool result = self->comm->send(buffer, size, dest_rank, tag);

  return PyBool_FromLong(result);
}

static PyObject* PyCommunicator_recv(PyCommunicator* self, PyObject* args) {
  int source_rank, tag;
  Py_ssize_t size;

  if (!PyArg_ParseTuple(args, "iin", &source_rank, &tag, &size)) {
    return NULL;
  }

  std::vector<char> buffer(size);

  bool result = self->comm->recv(buffer.data(), size, source_rank, tag);

  if (result) {
    return PyBytes_FromStringAndSize(buffer.data(), size);
  }

  Py_RETURN_NONE;
}

static PyObject* PyCommunicator_barrier(PyCommunicator* self, PyObject* args) {
  bool result = self->comm->barrier();
  return PyBool_FromLong(result);
}

static PyMethodDef PyCommunicator_methods[] = {
  {"initialize", (PyCFunction)PyCommunicator_initialize, METH_VARARGS,
   "Initialize the communicator"},
  {"allreduce", (PyCFunction)PyCommunicator_allreduce, METH_VARARGS,
   "Perform allreduce operation"},
  {"broadcast", (PyCFunction)PyCommunicator_broadcast, METH_VARARGS,
   "Perform broadcast operation"},
  {"send", (PyCFunction)PyCommunicator_send, METH_VARARGS,
   "Send message to rank"},
  {"recv", (PyCFunction)PyCommunicator_recv, METH_VARARGS,
   "Receive message from rank"},
  {"barrier", (PyCFunction)PyCommunicator_barrier, METH_NOARGS,
   "Synchronize all processes"},
  {NULL}
};

static PyTypeObject PyCommunicatorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "pccl.network.Communicator",
  sizeof(PyCommunicator),
  0,
  (destructor)PyCommunicator_dealloc,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  Py_TPFLAGS_DEFAULT,
  "PCCL Communicator",
  0, 0, 0, 0, 0, 0,
  PyCommunicator_methods,
  0, 0, 0, 0, 0, 0, 0,
  PyCommunicator_new,
};

static PyObject* create_connection(PyObject* self, PyObject* args) {
  const char* type = "tcp";

  if (!PyArg_ParseTuple(args, "|s", &type)) {
    return NULL;
  }

  engine_c::network::NetworkType network_type = engine_c::network::NetworkType::TCP_SOCKET;
  if (std::string(type) == "rdma") {
    network_type = engine_c::network::NetworkType::RDMA_VERBS;
  }

  engine_c::network::NetworkManager::getInstance().initialize(network_type);
  auto conn = engine_c::network::NetworkManager::getInstance().createConnection();

  if (!conn) {
    Py_RETURN_NONE;
  }

  PyNetworkConnection* py_conn = (PyNetworkConnection*)PyNetworkConnectionType.tp_alloc(&PyNetworkConnectionType, 0);
  if (py_conn) {
    new (&py_conn->conn) engine_c::network::ConnectionPtr(conn);
  }

  return (PyObject*)py_conn;
}

static PyObject* create_communicator(PyObject* self, PyObject* args) {
  const char* type = "tcp";

  if (!PyArg_ParseTuple(args, "|s", &type)) {
    return NULL;
  }

  engine_c::network::NetworkType network_type = engine_c::network::NetworkType::TCP_SOCKET;
  if (std::string(type) == "rdma") {
    network_type = engine_c::network::NetworkType::RDMA_VERBS;
  }

  auto comm = engine_c::communication::createCommunicator(network_type);

  if (!comm) {
    Py_RETURN_NONE;
  }

  PyCommunicator* py_comm = (PyCommunicator*)PyCommunicatorType.tp_alloc(&PyCommunicatorType, 0);
  if (py_comm) {
    new (&py_comm->comm) std::unique_ptr<engine_c::communication::Communicator>(comm);
  }

  return (PyObject*)py_comm;
}

static PyMethodDef pccl_network_methods[] = {
  {"create_connection", create_connection, METH_VARARGS,
   "Create a new network connection"},
  {"create_communicator", create_communicator, METH_VARARGS,
   "Create a new communicator"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pccl_network_module = {
  PyModuleDef_HEAD_INIT,
  "pccl.network",
  "PCCL Network Communication Module",
  -1,
  pccl_network_methods
};

PyMODINIT_FUNC PyInit_network(void) {
  PyObject* m;

  if (PyType_Ready(&PyNetworkConnectionType) < 0) {
    return NULL;
  }

  if (PyType_Ready(&PyCommunicatorType) < 0) {
    return NULL;
  }

  m = PyModule_Create(&pccl_network_module);
  if (m == NULL) {
    return NULL;
  }

  Py_INCREF(&PyNetworkConnectionType);
  PyModule_AddObject(m, "NetworkConnection", (PyObject*)&PyNetworkConnectionType);

  Py_INCREF(&PyCommunicatorType);
  PyModule_AddObject(m, "Communicator", (PyObject*)&PyCommunicatorType);

  return m;
}

}