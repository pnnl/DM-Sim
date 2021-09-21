#pragma once
#include <vector>
#include <memory>

namespace DmSim {
enum class OP {
  U3,
  U2,
  U1,
  CX,
  ID,
  X,
  Y,
  Z,
  H,
  S,
  SDG,
  T,
  TDG,
  RX,
  RY,
  RZ,
  CZ,
  CY,
  SWAP,
  CH,
  CCX,
  CSWAP,
  CRX,
  CRY,
  CRZ,
  CU1,
  CU3,
  RXX,
  RZZ,
  RCCX,
  RC3X,
  C3X,
  C3SQRTX,
  C4X,
  R,
  SRN,
  W,
  RYY
};

class DmSimBackend {
public:
  virtual void init(int n_qubits, int n_gpus = 1) = 0;
  virtual void addGate(OP op, const std::vector<int> &qubits,
                       const std::vector<double> &params = {}) = 0;
  virtual std::vector<int64_t> measure(int shots) = 0;
  virtual void finalize() = 0;
};

#ifdef XACC_HAS_CUDA
// Provided at by linking in the CUDA implementation
// TODO: use CppMicroservices registry (fixing nvcc compiling with Identifiable)
extern std::shared_ptr<DmSimBackend> getGpuDmSim();
#else
inline std::shared_ptr<DmSimBackend> getGpuDmSim() { return nullptr;}
#endif
} // namespace DmSim
