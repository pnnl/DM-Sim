#include <gtest/gtest.h>
#include "xacc.hpp"
#include "Optimizer.hpp"
#include "xacc_service.hpp"
#include "Algorithm.hpp"
#include "xacc_observable.hpp"
#include <fstream>

TEST(DmSimAcceleratorMultipleGpusTester, testMultipleGPUs) {
  auto accelerator =
      xacc::getAccelerator("dm-sim", {{"gpus", 4}, {"shots", 1024}});
  auto staqCompiler = xacc::getCompiler("staq");

  const std::string QASM_SRC_FILE =
      std::string(QASM_SOURCE_DIR) + "/qasm_src.txt";
  // Read source file:
  std::ifstream inFile;
  inFile.open(QASM_SRC_FILE);
  std::stringstream strStream;
  strStream << inFile.rdbuf();
  const std::string qasmSrcStr = strStream.str();

  // Compile:
  xacc::ScopeTimer timer("compile", false);
  auto IR = staqCompiler->compile(qasmSrcStr);
  auto program = IR->getComposites()[0];

  auto buffer1 = xacc::qalloc(10);
  accelerator->execute(buffer1, program);
  buffer1->print();
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  const auto result = RUN_ALL_TESTS();
  xacc::Finalize();
  return result;
}
