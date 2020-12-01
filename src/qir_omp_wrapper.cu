// ---------------------------------------------------------------------------
// DM-Sim: Density-Matrix Quantum Circuit Simulation Environment
// Version 2.2
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------
// File: qir_omp_wrapper.cpp
// DMSim interface to Q# and QIR.
// ---------------------------------------------------------------------------

#include <exception>
#include <iostream>
#include <stdexcept>
#include "config.hpp"
#include "SimulatorStub.hpp"

#ifdef NVGPU
#include "util.cuh"
#include "dmsim_nvgpu_omp.cuh"
#else
#include "util_cpu.h"
#include "dmsim_cpu_omp.hpp"
#endif

using namespace DMSim;

constexpr double fraction_to_theta(int numerator, int power) 
{
  return -1.0 * PI * numerator / (1ul << (power - 1));
}

class DM_QirSimulator final : public Microsoft::Quantum::SimulatorStub
{
 public:
  // The clients should never attempt to derefenece the Result, so we'll use
  // fake pointers to avoid allocation and deallocation.
  Result zero = reinterpret_cast<Result>(0xface0000);
  Result one = reinterpret_cast<Result>(0xface1000);
  char* qbase = reinterpret_cast<char*>(0xfce20000);

  IdxType to_qubit(QUBIT* Q) 
  {
      CHECK_NULL_POINTER(Q);
      IdxType q = static_cast<IdxType>(reinterpret_cast<char*>(Q) - qbase);
      return q;
  }
  QUBIT* from_qubit(IdxType qubit) 
  {
      return reinterpret_cast<QUBIT*>(qbase + qubit);
  }
  DM_QirSimulator():sim(NULL),last_cir(""),timer() 
  {
      n_qubits = 12;
      n_ranks = 4;
      is_new_sim = true;
      new_qubit_pos = 0;
      total_sim_time = 0;
      res_dm = NULL;
      sim = new Simulation(n_qubits, n_ranks);
  }
  ~DM_QirSimulator()
  {
      new_qubit_pos = 0;
      delete sim;
  }
  // Qubit management
  QUBIT* AllocateQubit() 
  {
      assert(new_qubit_pos < n_qubits);
      return from_qubit(new_qubit_pos++);
  }
  void ReleaseQubit(QUBIT* Q) 
  {
      //We currently do not actually release a qubit
      new_qubit_pos--;
      if (new_qubit_pos == 0)
      {
          is_new_sim = true;
          sim->clear_circuit();
      }
  }

  // Shortcuts
  void CX(QUBIT* Qcontrol, QUBIT* Qtarget) 
  {
      sim->append(Simulation::CX(to_qubit(Qcontrol), to_qubit(Qtarget)));
  }
  void CY(QUBIT* Qcontrol, QUBIT* Qtarget)
  {
      sim->append(Simulation::CY(to_qubit(Qcontrol), to_qubit(Qtarget)));
  }
  void CZ(QUBIT* Qcontrol, QUBIT* Qtarget)
  {
      sim->append(Simulation::CZ(to_qubit(Qcontrol), to_qubit(Qtarget)));
  }

  // Elementary operations
  void X(QUBIT* Qtarget) 
  {
      sim->append(Simulation::X(to_qubit(Qtarget)));
  }
  void Y(QUBIT* Qtarget)
  {
      sim->append(Simulation::Y(to_qubit(Qtarget)));
  }
  void Z(QUBIT* Qtarget)
  {
      sim->append(Simulation::Z(to_qubit(Qtarget)));
  }
  void H(QUBIT* Qtarget)
  {
      sim->append(Simulation::H(to_qubit(Qtarget)));
  }
  void S(QUBIT* Qtarget)
  {
      sim->append(Simulation::S(to_qubit(Qtarget)));
  }
  void T(QUBIT* Qtarget) 
  {
      sim->append(Simulation::T(to_qubit(Qtarget)));
  }
  void SWAP(QUBIT* Qtarget0, QUBIT* Qtarget1) 
  {
      sim->append(Simulation::SWAP(to_qubit(Qtarget0), to_qubit(Qtarget1)));
  }
  void Unitary(long numTargets, double** unitary, QUBIT* targets[]) 
  {
      throw std::logic_error("Unitary not_implemented");
  }
  void R(PauliId axis, QUBIT* Qtarget, double theta) 
  {
      IdxType target = to_qubit(Qtarget);
      switch (axis) 
      {
          case PauliId_I:
              //According to Microsoft "The Prelude" description, "Rotating 
              //around PauliI simply applies a global phase of theta/2".
              //Therefore, we just apply R(theta/2).
              sim->append(Simulation::R(theta/2.,target));
              break;
          case PauliId_X:
              sim->append(Simulation::RX(theta, target));
              break;
          case PauliId_Y:
              sim->append(Simulation::RY(theta, target));
              break;
          case PauliId_Z:
              sim->append(Simulation::RZ(theta, target));
              break;
      }
  }
  void RFraction(PauliId axis, QUBIT* Qtarget, long numerator, long power)
  {
      IdxType target = to_qubit(Qtarget);
      switch (axis) 
      {
          case PauliId_I:
              //According to Microsoft "The Prelude" description, "Rotating 
              //around PauliI simply applies a global phase of theta/2".
              //Therefore, we just apply R(theta/2).
              sim->append(Simulation::R(fraction_to_theta(numerator,power)/2.,target));
              break;
          case PauliId_X:
              sim->append(Simulation::RX(fraction_to_theta(numerator,power), target));
              break;
          case PauliId_Y:
              sim->append(Simulation::RY(fraction_to_theta(numerator,power), target));
              break;
          case PauliId_Z:
              sim->append(Simulation::RZ(fraction_to_theta(numerator, power), target));
              break;
      }
  }
  void R1(QUBIT* target, double theta)
  {
      //According to: https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.r1, R1 is equivalent to R(PauliZ,theta,qubit); R(PauliI,-theta,qubit); 
      R(PauliId_Z, target, theta);
      R(PauliId_I, target, -theta);
  }
  void R1Fraction(QUBIT* target, long numerator, long power) 
  {
      R1(target, fraction_to_theta(numerator, power));
  }
  void Exp(long numTargets, PauliId paulis[], QUBIT* targets[], double theta) 
  {
      //Currently we only have RXX, RYY and RZZ
      assert(numTargets == 2 && paulis[0] == paulis[1]);
      IdxType target0 = to_qubit(targets[0]);
      IdxType target1 = to_qubit(targets[1]);
      
      switch (paulis[0]) 
      {
          case PauliId_I:
              throw std::logic_error("RII not_implemented");
              //break;
          case PauliId_X:
              sim->append(Simulation::RXX(theta, target0, target1));
              break;
          case PauliId_Y:
              sim->append(Simulation::RYY(theta, target0, target1));
              break;
          case PauliId_Z:
              sim->append(Simulation::RZZ(theta, target0, target1));
              break;
      }
  }

  void ExpFraction(long numTargets, PauliId paulis[], QUBIT* targets[], long numerator, long power) 
  {
      Exp(numTargets, paulis, targets, fraction_to_theta(numerator, power));
  }
  void ControlledX(long numControls, QUBIT* Qcontrols[], QUBIT* Qtarget) 
  {
      //Currently we only have CX, CCX, C3X, C4X
      assert(numControls>0 && numControls<5);
      IdxType target = to_qubit(Qtarget);
      switch (numControls) 
      {
          case 1:
              sim->append(Simulation::CX(to_qubit(Qcontrols[0]), target));
              break;
          case 2:
              sim->append(Simulation::CCX(to_qubit(Qcontrols[0]),
                          to_qubit(Qcontrols[1]), target));
              break;
          case 3:
              sim->append(Simulation::C3X(to_qubit(Qcontrols[0]),
                          to_qubit(Qcontrols[1]), to_qubit(Qcontrols[2]),
                          target));
              break;
          case 4:
              sim->append(Simulation::C4X(to_qubit(Qcontrols[0]),
                          to_qubit(Qcontrols[1]), to_qubit(Qcontrols[2]),
                          to_qubit(Qcontrols[3]), target));
              break;
      }
  }
  void ControlledY(long numControls, QUBIT* Qcontrols[], QUBIT* Qtarget) 
  {
      //Currently we only have CY
      assert(numControls == 1);
      sim->append(Simulation::CY(to_qubit(Qcontrols[0]), to_qubit(Qtarget)));
  }
  void ControlledZ(long numControls, QUBIT* Qcontrols[], QUBIT* Qtarget) 
  {
      //Currently we only have CZ
      assert(numControls == 1);
      sim->append(Simulation::CZ(to_qubit(Qcontrols[0]), to_qubit(Qtarget)));
  }
  void ControlledH(long numControls, QUBIT* Qcontrols[], QUBIT* Qtarget) 
  {
      //Currently we only have CH
      assert(numControls == 1);
      sim->append(Simulation::CH(to_qubit(Qcontrols[0]), to_qubit(Qtarget)));
  }
  void ControlledS(long numControls, QUBIT* Qcontrols[], QUBIT* Qtarget) 
  {
      //S gate is phase shift by PI/2, so we call CU1 (controlled phase rotation)
      //by PI/2, currently we only support 1 control qubit
      assert(numControls == 1);
      sim->append(Simulation::CU1(PI/2, to_qubit(Qcontrols[0]), to_qubit(Qtarget)));
  }
  void ControlledT(long numControls, QUBIT* Qcontrols[], QUBIT* Qtarget) 
  {
      //T gate is phase shift by PI/4, so we call CU1 (controlled phase rotation)
      //by PI/4, currently we only support 1 control qubit
      assert(numControls == 1);
      sim->append(Simulation::CU1(PI/4, to_qubit(Qcontrols[0]), to_qubit(Qtarget)));
  }
  void ControlledSWAP(long numControls, QUBIT* Qcontrols[], QUBIT* Qtarget1, QUBIT* Qtarget2) 
  {
      //currently we only support 1 control qubit
      assert(numControls == 1);
      sim->append(Simulation::CSWAP(to_qubit(Qcontrols[0]), to_qubit(Qtarget1),
                  to_qubit(Qtarget2)));
  }
  void ControlledUnitary(long numControls, QUBIT* controls[], long numTargets,
                         double** unitary, QUBIT* targets[]) 
  {
      //we have CU3
      throw std::logic_error("ControlledUnitary not_implemented");
  }
  void ControlledR(long numControls, QUBIT* Qcontrols[], PauliId axis,
                   QUBIT* Qtarget, double theta) 
  {
      //currently we only support 1 control qubit
      assert(numControls == 1);
      IdxType target = to_qubit(Qtarget);
      IdxType control = to_qubit(Qcontrols[0]);
      switch (axis) 
      {
          case PauliId_I:
              throw std::logic_error("CRI not_implemented");
              //break;
          case PauliId_X:
              sim->append(Simulation::CRX(theta, control, target));
              break;
          case PauliId_Y:
              sim->append(Simulation::CRY(theta, control, target));
              break;
          case PauliId_Z:
              sim->append(Simulation::CRZ(theta, control, target));
              break;
      }
  }
  void ControlledRFraction(long numControls, QUBIT* Qcontrols[], PauliId axis,
                           QUBIT* Qtarget, long numerator, long power) 
  {
      //currently we only support 1 control qubit
      assert(numControls == 1);
      ControlledR(numControls, Qcontrols, axis, Qtarget,  
              fraction_to_theta(numerator, power));
  }
  void ControlledR1(long numControls, QUBIT* Qcontrols[], QUBIT* Qtarget, double theta) 
  {
      //currently we only support 1 control qubit
      assert(numControls == 1);
      //!!! The following might be wrong: R1=RZ+RI may not imply CR1=CRZ+CRI
      //According to: https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.r1, R1 is equivalent to R(PauliZ,theta,qubit); R(PauliI,-theta,qubit); 

      ControlledR(numControls, Qcontrols, PauliId_Z, Qtarget, theta);
      ControlledR(numControls, Qcontrols, PauliId_I, Qtarget, -theta);
  }
  void ControlledR1Fraction(long numControls, QUBIT* Qcontrols[],
          QUBIT* Qtarget, long numerator, long power) 
  {
      //currently we only support 1 control qubit
      assert(numControls == 1);
      ControlledR1(numControls, Qcontrols, Qtarget,
              fraction_to_theta(numerator, power));
  }
  void ControlledExp(long numControls, QUBIT* Qcontrols[], long numTargets,
          PauliId paulis[], QUBIT* Qtargets[], double theta) 
  {
      throw std::logic_error("ControlledExp not_implemented");
  }
  void ControlledExpFraction( long numControls, QUBIT* Qcontrols[], long numTargets, PauliId paulis[],
        QUBIT* Qtargets[], long numerator, long power) 
  {
      throw std::logic_error("ControlledExpFraction not_implemented");
  }

  void SAdjoint(QUBIT* Qtarget) 
  {
      sim->append(Simulation::SDG(to_qubit(Qtarget)));
  }
  void TAdjoint(QUBIT* Qtarget) 
  {
      sim->append(Simulation::TDG(to_qubit(Qtarget)));
  }
  void ControlledSAdjoint(long numControls, QUBIT* Qcontrols[],
          QUBIT* Qtarget) 
  {
      throw std::logic_error("Controlled SDG not_implemented");
  }
  void ControlledTAdjoint(long numControls, QUBIT* Qcontrols[],
          QUBIT* Qtarget) 
  {
      throw std::logic_error("Controlled TDG not_implemented");
  }

  void CustomOperation(long operationId, long numTargets,
          QUBIT* targets[], long numPaulis,
          PauliId paulis[], long numLongs, long longs[],
          long numDoubles, double doubles[], long numBools,
          bool bools[], long numResults, Result* results[],
          long numStrings, char** strings) 
  {
      throw std::logic_error("CustomOperation not_implemented");
  }
  void ControlledCustomOperation(
          long operationId, long numControls, QUBIT* controls[], long numTargets,
          QUBIT* targets[], long numPaulis, PauliId paulis[], long numLongs,
          long longs[], long numDoubles, double doubles[], long numBools,
          bool bools[], long numResults, Result* results[], long numStrings,
          char** strings) 
  {
      throw std::logic_error("ControlledCustomOperation not_implemented");
  }

  bool Assert(long numTargets, PauliId bases[], QUBIT* targets[],
          Result result, const char* failureMessage) 
  {
      return false;  // no-op
  }
  bool AssertProbability(long numTargets, PauliId bases[],
          QUBIT* targets[], double probabilityOfZero,
          double precision, const char* failureMessage) 
  {
      return false;  // no-op
  }
  // Results
  Result M(QUBIT* Qtarget) 
  {
      if (is_new_sim)
      {
          is_new_sim = false;
          std::string curr_cir = sim->dump();
          if (curr_cir != last_cir)
          {
              last_cir = curr_cir;
              sim->reset_dm();
              sim->upload();
              timer.start_timer(); 
              sim->sim();
              timer.start_timer();
              total_sim_time += timer.measure();
          }
          res_dm = sim->measure(1);
      }
      bool b_val = ((res_dm[0] >> to_qubit(Qtarget)) & 0x1);
      return reinterpret_cast<Result>(b_val);
  }
  Result Measure(long numBases, PauliId bases[], long numTargets,
          QUBIT* targets[]) 
  {
      throw std::logic_error("Measure not_implemented");
  }
  void Reset(QUBIT* Qtarget) 
  {
      throw std::logic_error("Reset not_implemented");
  }
  void ReleaseResult(Result result) 
  {
      // throw std::logic_error("not_implemented");
      return; // no-op
  }
  bool AreEqualResults(Result r1, Result r2) {
      return r1 == r2 ? true : false;
  }
  ResultValue GetResultValue(Result result) {
      return (result == one) ? Result_One : Result_Zero;
  }
  Result UseZero() { return zero; }
  Result UseOne() { return one; }

 private:
  Simulation* sim;
  std::string last_cir;
  unsigned n_qubits;
  unsigned n_ranks;
  bool is_new_sim;
  IdxType* res_dm;
  IdxType new_qubit_pos;
  double total_sim_time;
  cpu_timer timer;
};

extern "C" Microsoft::Quantum::ISimulator* GetDMSim() 
{
    static Microsoft::Quantum::ISimulator* g_iqa = nullptr;
    if(!g_iqa) {
        g_iqa = new DM_QirSimulator{};
        // ResultOne = g_iqa->UseOne();
        // ResultZero = g_iqa->UseZero();
    }
    return g_iqa;
}

