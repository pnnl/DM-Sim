#include "DmSimAccelerator.hpp"
#include "DmSimApi.hpp"
#include "AllGateVisitor.hpp"
#include "xacc_plugin.hpp"
#include <thread>
namespace xacc {
namespace quantum {
class DmSimCircuitVisitor : public AllGateVisitor {
private:
  DmSim::DmSimBackend *m_backend;

public:
  DmSimCircuitVisitor(DmSim::DmSimBackend *backend) : m_backend(backend) {}

  void visit(Hadamard &h) override {
    m_backend->addGate(DmSim::OP::H, std::vector<int>{(int)h.bits()[0]});
  }

  void visit(X &x) override {
    m_backend->addGate(DmSim::OP::X, std::vector<int>{(int)x.bits()[0]});
  }

  void visit(Y &y) override {
    m_backend->addGate(DmSim::OP::Y, std::vector<int>{(int)y.bits()[0]});
  }

  void visit(Z &z) override {
    m_backend->addGate(DmSim::OP::Z, std::vector<int>{(int)z.bits()[0]});
  }

  void visit(Rx &rx) override {
    m_backend->addGate(DmSim::OP::RX, std::vector<int>{(int)rx.bits()[0]},
                       {InstructionParameterToDouble(rx.getParameter(0))});
  }

  void visit(Ry &ry) override {
    m_backend->addGate(DmSim::OP::RY, std::vector<int>{(int)ry.bits()[0]},
                       {InstructionParameterToDouble(ry.getParameter(0))});
  }

  void visit(Rz &rz) override {
    m_backend->addGate(DmSim::OP::RZ, std::vector<int>{(int)rz.bits()[0]},
                       {InstructionParameterToDouble(rz.getParameter(0))});
  }

  void visit(S &s) override {
    m_backend->addGate(DmSim::OP::S, std::vector<int>{(int)s.bits()[0]});
  }

  void visit(Sdg &sdg) override {
    m_backend->addGate(DmSim::OP::SDG, std::vector<int>{(int)sdg.bits()[0]});
  }

  void visit(T &t) override {
    m_backend->addGate(DmSim::OP::T, std::vector<int>{(int)t.bits()[0]});
  }

  void visit(Tdg &tdg) override {
    m_backend->addGate(DmSim::OP::TDG, std::vector<int>{(int)tdg.bits()[0]});
  }

  void visit(CNOT &cnot) override {
    m_backend->addGate(DmSim::OP::CX, std::vector<int>{(int)cnot.bits()[0],
                                                       (int)cnot.bits()[1]});
  }

  void visit(CY &cy) override {
    m_backend->addGate(DmSim::OP::CY,
                       std::vector<int>{(int)cy.bits()[0], (int)cy.bits()[1]});
  }

  void visit(CZ &cz) override {
    m_backend->addGate(DmSim::OP::CZ,
                       std::vector<int>{(int)cz.bits()[0], (int)cz.bits()[1]});
  }

  void visit(Swap &s) override {
    m_backend->addGate(DmSim::OP::SWAP,
                       std::vector<int>{(int)s.bits()[0], (int)s.bits()[1]});
  }

  void visit(CH &ch) override {
    m_backend->addGate(DmSim::OP::CH,
                       std::vector<int>{(int)ch.bits()[0], (int)ch.bits()[1]});
  }

  void visit(CPhase &cphase) override {
    m_backend->addGate(
        DmSim::OP::CU1,
        std::vector<int>{(int)cphase.bits()[0], (int)cphase.bits()[1]},
        {InstructionParameterToDouble(cphase.getParameter(0))});
  }

  void visit(CRZ &crz) override {
    m_backend->addGate(DmSim::OP::CRZ,
                       std::vector<int>{(int)crz.bits()[0], (int)crz.bits()[1]},
                       {InstructionParameterToDouble(crz.getParameter(0))});
  }

  void visit(Identity &i) override {}

  void visit(U &u) override {
    const auto theta = InstructionParameterToDouble(u.getParameter(0));
    const auto phi = InstructionParameterToDouble(u.getParameter(1));
    const auto lambda = InstructionParameterToDouble(u.getParameter(2));

    m_backend->addGate(DmSim::OP::U3, std::vector<int>{(int)u.bits()[0]},
                       {theta, phi, lambda});
  }

  void visit(Measure &measure) override {
    m_measureQubits.emplace_back(measure.bits()[0]);
  }

  // NOT SUPPORTED:
  void visit(IfStmt &ifStmt) override {}
  std::vector<size_t> getMeasureBits() const { return m_measureQubits; }

private:
  std::vector<size_t> m_measureQubits;
};

void DmSimAccelerator::initialize(const HeterogeneousMap &params) {
#ifdef XACC_HAS_CUDA
  m_backend = "gpu";
#else
  m_backend = "cpu";
#endif
  if (params.stringExists("backend")) {
    m_backend = params.getString("backend");
  }
  m_processingUnits = 1;
  if (m_backend == "gpu") {
    if (params.keyExists<int>("gpus")) {
      m_processingUnits = params.get<int>("gpus");
    }
  }
  if (m_backend == "cpu") {
    // Default is to use all threads
    const auto getNearestPowerOf2 = [](int num) {
      int result = 1;
      while (result <= num) {
        result *= 2;
      }
      return result;
    };
    const auto num_threads = std::thread::hardware_concurrency() != 0
                                 ? std::thread::hardware_concurrency()
                                 : 1;
    m_processingUnits = getNearestPowerOf2(num_threads);
    if (params.keyExists<int>("threads")) {
      m_processingUnits = params.get<int>("threads");
    }
  }

  m_shots = 1024;
  if (params.keyExists<int>("shots")) {
    m_shots = params.get<int>("shots");
  }
}

std::shared_ptr<DmSim::DmSimBackend> DmSimAccelerator::get_backend() {
  if (m_backend == "gpu") {
    auto ptr = DmSim::getGpuDmSim();
    assert(ptr);
    return ptr;
  }
  return DmSim::getCpuDmSim();
}

void DmSimAccelerator::execute(
    std::shared_ptr<AcceleratorBuffer> buffer,
    const std::shared_ptr<CompositeInstruction> compositeInstruction) {
  auto dm_sim = get_backend();
  if (!dm_sim) {
    xacc::error("DM-Sim was not installed. Please make sure that you're "
                "compiling XACC on a platform with CUDA.");
  }
  dm_sim->init(buffer->size(),
               std::min(m_processingUnits, 1 << buffer->size()));
  DmSimCircuitVisitor visitor(dm_sim.get());
  // Walk the IR tree, and visit each node
  InstructionIterator it(compositeInstruction);
  while (it.hasNext()) {
    auto nextInst = it.next();
    if (nextInst->isEnabled()) {
      nextInst->accept(&visitor);
    }
  }
  auto measured_bits = visitor.getMeasureBits();
  if (measured_bits.empty()) {
    // Default is just measure alls:
    for (size_t i = 0; i < buffer->size(); ++i) {
      measured_bits.emplace_back(i);
    }
  }
  std::sort(measured_bits.begin(), measured_bits.end());
  const auto measured_results = dm_sim->measure(m_shots);
  const auto dmSimMeasureToBitString = [&measured_bits](const auto &val) {
    std::string bitString;
    for (const auto &bit : measured_bits) {
      if (val & (1ULL << bit)) {
        bitString.push_back('1');
      } else {
        bitString.push_back('0');
      }
    }
    return bitString;
  };
  for (const auto &m : measured_results) {
    buffer->appendMeasurement(dmSimMeasureToBitString(m));
  }
}
void DmSimAccelerator::execute(
    std::shared_ptr<AcceleratorBuffer> buffer,
    const std::vector<std::shared_ptr<CompositeInstruction>>
        compositeInstructions) {
  for (auto &f : compositeInstructions) {
    auto tmpBuffer =
        std::make_shared<xacc::AcceleratorBuffer>(f->name(), buffer->size());
    execute(tmpBuffer, f);
    buffer->appendChild(f->name(), tmpBuffer);
  }
}
} // namespace quantum
} // namespace xacc
REGISTER_ACCELERATOR(xacc::quantum::DmSimAccelerator)