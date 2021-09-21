#pragma one
#include "xacc.hpp"

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
  int m_ngpus;
  int m_shots;
};
} // namespace quantum
} // namespace xacc
