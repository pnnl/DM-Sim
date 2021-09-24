#pragma one
#include "xacc.hpp"
namespace DmSim {
// Forward declare
class DmSimBackend;
} // namespace DmSim
namespace xacc {
namespace quantum {
class DmSimAccelerator : public Accelerator {
public:
  // Identifiable interface impls
  virtual const std::string name() const override { return "dm-sim"; }
  virtual const std::string description() const override {
    return "XACC Simulation Accelerator based on DM-Sim library.";
  }

  // Accelerator interface impls
  virtual void initialize(const HeterogeneousMap &params = {}) override;
  virtual void updateConfiguration(const HeterogeneousMap &config) override {
    initialize(config);
  };
  virtual const std::vector<std::string> configurationKeys() override {
    return {};
  }
  virtual BitOrder getBitOrder() override { return BitOrder::LSB; }
  virtual void execute(std::shared_ptr<AcceleratorBuffer> buffer,
                       const std::shared_ptr<CompositeInstruction>
                           compositeInstruction) override;
  virtual void execute(std::shared_ptr<AcceleratorBuffer> buffer,
                       const std::vector<std::shared_ptr<CompositeInstruction>>
                           compositeInstructions) override;

private:
  std::shared_ptr<DmSim::DmSimBackend> get_backend();
  int m_processingUnits;
  int m_shots;
  std::string m_backend;
};
} // namespace quantum
} // namespace xacc
