#include "../DmSimApi.hpp"
#include "src/util_nvgpu.cuh"
#include "src/dmsim_nvgpu_omp.cuh"
#include <memory>

namespace DmSim {
class NvidiaOmp : public DmSimBackend {
public:
  virtual void init(int n_qubits, int n_gpus) override {
    m_sim = std::make_shared<::DMSim::Simulation>(n_qubits, n_gpus);
  }
  virtual void addGate(OP op, const std::vector<int> &qubits,
                       const std::vector<double> &params) override {
    switch (op) {
    case OP::U3:
      assert(qubits.size() == 1);
      assert(params.size() == 3);
      m_sim->append(
          ::DMSim::Simulation::U3(params[0], params[1], params[2], qubits[0]));
      break;
    case OP::U2:
      assert(qubits.size() == 1);
      assert(params.size() == 2);
      m_sim->append(::DMSim::Simulation::U2(params[0], params[1], qubits[0]));
      break;
    case OP::U1:
      assert(qubits.size() == 1);
      assert(params.size() == 1);
      m_sim->append(::DMSim::Simulation::U1(params[0], qubits[0]));
      break;
    case OP::CX:
      assert(qubits.size() == 2);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::CX(qubits[0], qubits[1]));
      break;
    case OP::X:
      assert(qubits.size() == 1);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::X(qubits[0]));
      break;
    case OP::Y:
      assert(qubits.size() == 1);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::Y(qubits[0]));
      break;
    case OP::Z:
      assert(qubits.size() == 1);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::Z(qubits[0]));
      break;
    case OP::H:
      assert(qubits.size() == 1);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::H(qubits[0]));
      break;
    case OP::S:
      assert(qubits.size() == 1);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::S(qubits[0]));
      break;
    case OP::SDG:
      assert(qubits.size() == 1);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::SDG(qubits[0]));
      break;
    case OP::T:
      assert(qubits.size() == 1);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::T(qubits[0]));
      break;
    case OP::TDG:
      assert(qubits.size() == 1);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::TDG(qubits[0]));
      break;
    case OP::RX:
      assert(qubits.size() == 1);
      assert(params.size() == 1);
      m_sim->append(::DMSim::Simulation::RX(params[0], qubits[0]));
      break;
    case OP::RY:
      assert(qubits.size() == 1);
      assert(params.size() == 1);
      m_sim->append(::DMSim::Simulation::RY(params[0], qubits[0]));
      break;
    case OP::RZ:
      assert(qubits.size() == 1);
      assert(params.size() == 1);
      m_sim->append(::DMSim::Simulation::RZ(params[0], qubits[0]));
      break;
    case OP::CZ:
      assert(qubits.size() == 2);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::CZ(qubits[0], qubits[1]));
      break;
    case OP::CY:
      assert(qubits.size() == 2);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::CY(qubits[0], qubits[1]));
      break;
    case OP::SWAP:
      assert(qubits.size() == 2);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::SWAP(qubits[0], qubits[1]));
      break;
    case OP::CH:
      assert(qubits.size() == 2);
      assert(params.size() == 0);
      m_sim->append(::DMSim::Simulation::SWAP(qubits[0], qubits[1]));
      break;
    case OP::CRX:
      assert(qubits.size() == 2);
      assert(params.size() == 1);
      m_sim->append(::DMSim::Simulation::CRX(params[0], qubits[0], qubits[1]));
      break;
    case OP::CRY:
      assert(qubits.size() == 2);
      assert(params.size() == 1);
      m_sim->append(::DMSim::Simulation::CRY(params[0], qubits[0], qubits[1]));
      break;
    case OP::CRZ:
      assert(qubits.size() == 2);
      assert(params.size() == 1);
      m_sim->append(::DMSim::Simulation::CRZ(params[0], qubits[0], qubits[1]));
      break;
    case OP::CU1:
      assert(qubits.size() == 2);
      assert(params.size() == 1);
      m_sim->append(::DMSim::Simulation::CU1(params[0], qubits[0], qubits[1]));
      break;
    case OP::ID:
    case OP::CCX:
    case OP::CSWAP:
    case OP::CU3:
    case OP::RXX:
    case OP::RZZ:
    case OP::RCCX:
    case OP::RC3X:
    case OP::C3X:
    case OP::C3SQRTX:
    case OP::C4X:
    case OP::R:
    case OP::SRN:
    case OP::W:
    case OP::RYY:
    default:
      // Don't support
      __builtin_unreachable();
    }
  }
  virtual std::vector<int64_t> measure(int shots) override {
    // Upload to GPU, ready for execution
    m_sim->upload();

    // Run the simulation
    m_sim->sim();

    // Measure
    auto *res = m_sim->measure(shots);
    std::vector<int64_t> result;
    result.reserve(shots);
    for (int i = 0; i < shots; ++i) {
      result.emplace_back(res[i]);
    }
    delete res;
    return result;
  }

  virtual void finalize() override { m_sim.reset(); }

private:
  std::shared_ptr<::DMSim::Simulation> m_sim;
};
std::shared_ptr<DmSimBackend> getGpuDmSim() {
  // std::cout << "HOWDY: create DmSim runner\n";
  return std::make_shared<NvidiaOmp>();
}
} // namespace DmSim
