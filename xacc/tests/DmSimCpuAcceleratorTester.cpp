#include "xacc.hpp"
#include "xacc_observable.hpp"
#include "xacc_service.hpp"
#include <gtest/gtest.h>
#include <string>

namespace {
template <typename T> std::vector<T> linspace(T a, T b, size_t N) {
  T h = (b - a) / static_cast<T>(N - 1);
  std::vector<T> xs(N);
  typename std::vector<T>::iterator x;
  T val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
    *x = val;
  }
  return xs;
}
} // namespace

TEST(DmSimCpuAcceleratorTester, testDeuteron) {
  auto accelerator = xacc::getAccelerator("dm-sim:cpu", {{"shots", 100000}});
  auto xasmCompiler = xacc::getCompiler("xasm");
  auto ir = xasmCompiler->compile(R"(__qpu__ void ansatz(qbit q, double t) {
      X(q[0]);
      Ry(q[1], t);
      CX(q[1], q[0]);
      H(q[0]);
      H(q[1]);
      Measure(q[0]);
      Measure(q[1]);
    })",
                                  accelerator);

  auto program = ir->getComposite("ansatz");
  // Expected results from deuteron_2qbit_xasm_X0X1
  const std::vector<double> expectedResults{
      0.0,       -0.324699, -0.614213, -0.837166, -0.9694,
      -0.996584, -0.915773, -0.735724, -0.475947, -0.164595,
      0.164595,  0.475947,  0.735724,  0.915773,  0.996584,
      0.9694,    0.837166,  0.614213,  0.324699,  0.0};

  const auto angles = linspace(-xacc::constants::pi, xacc::constants::pi, 20);
  for (size_t i = 0; i < angles.size(); ++i) {
    auto buffer = xacc::qalloc(2);
    auto evaled = program->operator()({angles[i]});
    accelerator->execute(buffer, evaled);
    EXPECT_NEAR(buffer->getExpectationValueZ(), expectedResults[i], 0.01);
  }
}

TEST(DmSimCpuAcceleratorTester, testShots) {
  const int nbShots = 100;
  {
    auto accelerator =
        xacc::getAccelerator("dm-sim:cpu", {std::make_pair("shots", nbShots)});
    // Allocate some qubits
    auto buffer = xacc::qalloc(2);
    auto quilCompiler = xacc::getCompiler("quil");
    // Nothing, should be 00
    auto ir = quilCompiler->compile(R"(__qpu__ void test1(qbit q) {
MEASURE 0 [0]
MEASURE 1 [1]
})",
                                    accelerator);
    accelerator->execute(buffer, ir->getComposites()[0]);
    EXPECT_EQ(buffer->getMeasurementCounts().size(), 1);
    EXPECT_EQ(buffer->getMeasurementCounts()["00"], nbShots);
    buffer->print();
  }
  {
    auto accelerator =
        xacc::getAccelerator("dm-sim:cpu", {std::make_pair("shots", nbShots)});
    auto buffer = xacc::qalloc(2);
    auto quilCompiler = xacc::getCompiler("quil");
    // Expected "11"
    auto ir = quilCompiler->compile(R"(__qpu__ void test2(qbit q) {
X 0
CX 0 1
MEASURE 0 [0]
MEASURE 1 [1]
})",
                                    accelerator);
    accelerator->execute(buffer, ir->getComposites()[0]);

    EXPECT_EQ(buffer->getMeasurementCounts().size(), 1);
    EXPECT_EQ(buffer->getMeasurementCounts()["11"], nbShots);
  }
  {
    auto accelerator =
        xacc::getAccelerator("dm-sim:cpu", {std::make_pair("shots", nbShots)});
    auto buffer = xacc::qalloc(2);
    auto quilCompiler = xacc::getCompiler("quil");
    // Bell states
    auto ir = quilCompiler->compile(R"(__qpu__ void test3(qbit q) {
H 0
CX 0 1
MEASURE 0 [0]
MEASURE 1 [1]
})",
                                    accelerator);
    accelerator->execute(buffer, ir->getComposites()[0]);
    EXPECT_EQ(buffer->getMeasurementCounts().size(), 2);
    // Only 00 and 11 states
    EXPECT_EQ(buffer->getMeasurementCounts()["11"] +
                  buffer->getMeasurementCounts()["00"],
              nbShots);
  }
}

TEST(DmSimCpuAcceleratorTester, testDeuteronVqeH3Shots) {
  const int nbShots = 100000;
  auto accelerator = xacc::getAccelerator("dm-sim:cpu", {{"shots", nbShots}});

  // Create the N=3 deuteron Hamiltonian
  auto H_N_3 = xacc::quantum::getObservable(
      "pauli",
      std::string("5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1 + "
                  "9.625 - 9.625 Z2 - 3.91 X1 X2 - 3.91 Y1 Y2"));

  auto optimizer =
      xacc::getOptimizer("nlopt", {std::make_pair("nlopt-maxeval", 100)});

  // JIT map Quil QASM Ansatz to IR
  xacc::qasm(R"(
        .compiler xasm
        .circuit deuteron_ansatz_h3_2
        .parameters t0, t1
        .qbit q
        X(q[0]);
        exp_i_theta(q, t0, {{"pauli", "X0 Y1 - Y0 X1"}});
        exp_i_theta(q, t1, {{"pauli", "X0 Z1 Y2 - X2 Z1 Y0"}});
    )");
  auto ansatz = xacc::getCompiled("deuteron_ansatz_h3_2");

  // Get the VQE Algorithm and initialize it
  auto vqe = xacc::getAlgorithm("vqe");
  vqe->initialize({std::make_pair("ansatz", ansatz),
                   std::make_pair("observable", H_N_3),
                   std::make_pair("accelerator", accelerator),
                   std::make_pair("optimizer", optimizer)});

  xacc::set_verbose(true);
  // Allocate some qubits and execute
  auto buffer = xacc::qalloc(3);
  vqe->execute(buffer);

  // Expected result: -2.04482
  // Tol: 0.25 (~10% of the true value)
  // (since we are using shots, hence will introduce randomness to the
  // optimizer)
  std::cout << "Energy = " << (*buffer)["opt-val"].as<double>() << "\n";
  EXPECT_NEAR((*buffer)["opt-val"].as<double>(), -2.04482, 0.25);
}

int main(int argc, char **argv) {
  xacc::Initialize();

  ::testing::InitGoogleTest(&argc, argv);
  const auto result = RUN_ALL_TESTS();

  xacc::Finalize();

  return result;
}
