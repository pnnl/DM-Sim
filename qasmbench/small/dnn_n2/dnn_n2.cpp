#include <stdio.h>
#include "util_cpu.h"
#include "dmsim_cpu_omp.hpp"
//Use the DMSim namespace to enable C++/CUDA APIs
using namespace DMSim;

// A TWO QUBIT CIRCIUT 3 LAYERS DEEP
// Generated from Cirq v0.8.0
// Qubits: [(0, 0), (0, 1)]
void prepare_circuit(Simulation &sim)
{
	sim.append(Simulation::RX(1.09999999999, 0));
	sim.append(Simulation::RY(1.09999999999, 0));
	sim.append(Simulation::RZ(1.09999999999, 0));
	sim.append(Simulation::RX(1.09999999999, 1));
	sim.append(Simulation::RY(1.09999999999, 1));
	sim.append(Simulation::RZ(1.09999999999, 1));
// Gate: ZZ**1.1
	sim.append(Simulation::RZ(3.45575191895, 0));
	sim.append(Simulation::RZ(3.45575191895, 1));
	sim.append(Simulation::U3(1.57079632679, 0, 0.785398163397, 0));
	sim.append(Simulation::U3(1.57079632679, 3.14159265359, 2.35619449019, 1));
	sim.append(Simulation::RX(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(1.25663706144, 0));
	sim.append(Simulation::RY(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(-1.57079632679, 1));
	sim.append(Simulation::RZ(1.57079632679, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::U3(1.57079632679, 2.04203522483, 3.14159265359, 0));
	sim.append(Simulation::U3(1.57079632679, 0.471238898038, 0, 1));
// Gate: YY**1.1
	sim.append(Simulation::U3(0, 3.14159265359, 1.57079632679, 0));
	sim.append(Simulation::U3(0, 0, 1.57079632679, 1));
	sim.append(Simulation::RX(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(1.25663706144, 0));
	sim.append(Simulation::RY(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(-1.57079632679, 1));
	sim.append(Simulation::RZ(1.57079632679, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::U3(3.14159265359, 0, 1.57079632679, 0));
	sim.append(Simulation::U3(3.14159265359, 0, 4.71238898038, 1));
// Gate: XX**1.1
	sim.append(Simulation::U3(1.57079632679, 4.71238898038, 4.71238898038, 0));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 4.71238898038, 1));
	sim.append(Simulation::RX(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(1.25663706144, 0));
	sim.append(Simulation::RY(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(-1.57079632679, 1));
	sim.append(Simulation::RZ(1.57079632679, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 1.57079632679, 0));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 4.71238898038, 1));
	sim.append(Simulation::RX(1.09999999999, 0));
	sim.append(Simulation::RY(1.09999999999, 0));
	sim.append(Simulation::RZ(1.09999999999, 0));
	sim.append(Simulation::RX(1.09999999999, 1));
	sim.append(Simulation::RY(1.09999999999, 1));
	sim.append(Simulation::RZ(1.09999999999, 1));
	sim.append(Simulation::RX(1.09999999999, 1));
	sim.append(Simulation::RY(1.09999999999, 1));
	sim.append(Simulation::RZ(1.09999999999, 1));
	sim.append(Simulation::RX(1.09999999999, 0));
	sim.append(Simulation::RY(1.09999999999, 0));
	sim.append(Simulation::RZ(1.09999999999, 0));
// Gate: ZZ**1.1
	sim.append(Simulation::RZ(3.45575191895, 1));
	sim.append(Simulation::RZ(3.45575191895, 0));
	sim.append(Simulation::U3(1.57079632679, 0, 0.785398163397, 1));
	sim.append(Simulation::U3(1.57079632679, 3.14159265359, 2.35619449019, 0));
	sim.append(Simulation::RX(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(1.25663706144, 1));
	sim.append(Simulation::RY(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(-1.57079632679, 0));
	sim.append(Simulation::RZ(1.57079632679, 0));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::U3(1.57079632679, 2.04203522483, 3.14159265359, 1));
	sim.append(Simulation::U3(1.57079632679, 0.471238898038, 0, 0));
// Gate: YY**1.1
	sim.append(Simulation::U3(0, 3.14159265359, 1.57079632679, 1));
	sim.append(Simulation::U3(0, 0, 1.57079632679, 0));
	sim.append(Simulation::RX(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(1.25663706144, 1));
	sim.append(Simulation::RY(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(-1.57079632679, 0));
	sim.append(Simulation::RZ(1.57079632679, 0));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::U3(3.14159265359, 0, 1.57079632679, 1));
	sim.append(Simulation::U3(3.14159265359, 0, 4.71238898038, 0));
// Gate: XX**1.1
	sim.append(Simulation::U3(1.57079632679, 4.71238898038, 4.71238898038, 1));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 4.71238898038, 0));
	sim.append(Simulation::RX(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(1.25663706144, 1));
	sim.append(Simulation::RY(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(-1.57079632679, 0));
	sim.append(Simulation::RZ(1.57079632679, 0));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 1.57079632679, 1));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 4.71238898038, 0));
	sim.append(Simulation::RX(1.09999999999, 1));
	sim.append(Simulation::RY(1.09999999999, 1));
	sim.append(Simulation::RZ(1.09999999999, 1));
	sim.append(Simulation::RX(1.09999999999, 0));
	sim.append(Simulation::RY(1.09999999999, 0));
	sim.append(Simulation::RZ(1.09999999999, 0));
// Gate: CNOT**1.1
	sim.append(Simulation::RY(-1.57079632679, 1));
	sim.append(Simulation::U3(1.57079632679, 0, 0.785398163397, 0));
	sim.append(Simulation::U3(1.57079632679, 3.14159265359, 2.35619449019, 1));
	sim.append(Simulation::RX(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(0.157079632679, 0));
	sim.append(Simulation::RY(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(-1.57079632679, 1));
	sim.append(Simulation::RZ(1.57079632679, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::U3(1.57079632679, 0.942477796077, 3.14159265359, 0));
	sim.append(Simulation::U3(1.57079632679, 5.65486677646, 0, 1));
	sim.append(Simulation::RY(1.57079632679, 1));
// Gate: CZ**1.1
	sim.append(Simulation::U3(1.57079632679, 0, 0.785398163397, 0));
	sim.append(Simulation::U3(1.57079632679, 3.14159265359, 2.35619449019, 1));
	sim.append(Simulation::RX(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(0.157079632679, 0));
	sim.append(Simulation::RY(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(-1.57079632679, 1));
	sim.append(Simulation::RZ(1.57079632679, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::U3(1.57079632679, 0.942477796077, 3.14159265359, 0));
	sim.append(Simulation::U3(1.57079632679, 5.65486677646, 0, 1));
	sim.append(Simulation::RX(1.09999999999, 0));
	sim.append(Simulation::RY(1.09999999999, 0));
	sim.append(Simulation::RZ(1.09999999999, 0));
	sim.append(Simulation::RX(1.09999999999, 1));
	sim.append(Simulation::RY(1.09999999999, 1));
	sim.append(Simulation::RZ(1.09999999999, 1));
// Gate: ZZ**1.1
	sim.append(Simulation::RZ(3.45575191895, 0));
	sim.append(Simulation::RZ(3.45575191895, 1));
	sim.append(Simulation::U3(1.57079632679, 0, 0.785398163397, 0));
	sim.append(Simulation::U3(1.57079632679, 3.14159265359, 2.35619449019, 1));
	sim.append(Simulation::RX(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(1.25663706144, 0));
	sim.append(Simulation::RY(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(-1.57079632679, 1));
	sim.append(Simulation::RZ(1.57079632679, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::U3(1.57079632679, 2.04203522483, 3.14159265359, 0));
	sim.append(Simulation::U3(1.57079632679, 0.471238898038, 0, 1));
// Gate: YY**1.1
	sim.append(Simulation::U3(0, 3.14159265359, 1.57079632679, 0));
	sim.append(Simulation::U3(0, 0, 1.57079632679, 1));
	sim.append(Simulation::RX(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(1.25663706144, 0));
	sim.append(Simulation::RY(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(-1.57079632679, 1));
	sim.append(Simulation::RZ(1.57079632679, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::U3(3.14159265359, 0, 1.57079632679, 0));
	sim.append(Simulation::U3(3.14159265359, 0, 4.71238898038, 1));
// Gate: XX**1.1
	sim.append(Simulation::U3(1.57079632679, 4.71238898038, 4.71238898038, 0));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 4.71238898038, 1));
	sim.append(Simulation::RX(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(1.25663706144, 0));
	sim.append(Simulation::RY(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(-1.57079632679, 1));
	sim.append(Simulation::RZ(1.57079632679, 1));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 1.57079632679, 0));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 4.71238898038, 1));
	sim.append(Simulation::RX(1.09999999999, 0));
	sim.append(Simulation::RY(1.09999999999, 0));
	sim.append(Simulation::RZ(1.09999999999, 0));
	sim.append(Simulation::RX(1.09999999999, 1));
	sim.append(Simulation::RY(1.09999999999, 1));
	sim.append(Simulation::RZ(1.09999999999, 1));
	sim.append(Simulation::RX(1.09999999999, 1));
	sim.append(Simulation::RY(1.09999999999, 1));
	sim.append(Simulation::RZ(1.09999999999, 1));
	sim.append(Simulation::RX(1.09999999999, 0));
	sim.append(Simulation::RY(1.09999999999, 0));
	sim.append(Simulation::RZ(1.09999999999, 0));
// Gate: ZZ**1.1
	sim.append(Simulation::RZ(3.45575191895, 1));
	sim.append(Simulation::RZ(3.45575191895, 0));
	sim.append(Simulation::U3(1.57079632679, 0, 0.785398163397, 1));
	sim.append(Simulation::U3(1.57079632679, 3.14159265359, 2.35619449019, 0));
	sim.append(Simulation::RX(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(1.25663706144, 1));
	sim.append(Simulation::RY(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(-1.57079632679, 0));
	sim.append(Simulation::RZ(1.57079632679, 0));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::U3(1.57079632679, 2.04203522483, 3.14159265359, 1));
	sim.append(Simulation::U3(1.57079632679, 0.471238898038, 0, 0));
// Gate: YY**1.1
	sim.append(Simulation::U3(0, 3.14159265359, 1.57079632679, 1));
	sim.append(Simulation::U3(0, 0, 1.57079632679, 0));
	sim.append(Simulation::RX(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(1.25663706144, 1));
	sim.append(Simulation::RY(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(-1.57079632679, 0));
	sim.append(Simulation::RZ(1.57079632679, 0));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::U3(3.14159265359, 0, 1.57079632679, 1));
	sim.append(Simulation::U3(3.14159265359, 0, 4.71238898038, 0));
// Gate: XX**1.1
	sim.append(Simulation::U3(1.57079632679, 4.71238898038, 4.71238898038, 1));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 4.71238898038, 0));
	sim.append(Simulation::RX(1.57079632679, 1));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::RX(1.25663706144, 1));
	sim.append(Simulation::RY(1.57079632679, 0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::RX(-1.57079632679, 0));
	sim.append(Simulation::RZ(1.57079632679, 0));
	sim.append(Simulation::CX(1, 0));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 1.57079632679, 1));
	sim.append(Simulation::U3(1.57079632679, 1.57079632679, 4.71238898038, 0));
	sim.append(Simulation::RX(1.09999999999, 1));
	sim.append(Simulation::RY(1.09999999999, 1));
	sim.append(Simulation::RZ(1.09999999999, 1));
	sim.append(Simulation::RX(1.09999999999, 0));
	sim.append(Simulation::RY(1.09999999999, 0));
	sim.append(Simulation::RZ(1.09999999999, 0));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=2;
	int n_cpus=256;
	Simulation sim(n_qubits, n_cpus);
	prepare_circuit(sim);
	sim.upload();
	sim.sim();
	auto* res = sim.measure(5);
	print_measurement(res, n_qubits, 5);
	delete res; 
	return 0;
}
