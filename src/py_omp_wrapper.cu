// ---------------------------------------------------------------------------
// File: dmsim_sin.cuh
// Python wrapper for the OpenMP implementation 
//    (single-node-multi-GPUs) of DM-Sim. 
// ---------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "config.hpp"
#include "util.cuh"
#include "gate_omp.cuh"

namespace py = pybind11;
using namespace DMSim;

//Note pybind11 requires CUDA-10.0 or newer
PYBIND11_MODULE(dmsim_py_omp_wrapper, m) 
{
    py::class_<Gate>(m, "Gate")
        .def(py::init<enum OP,IdxType,IdxType,IdxType,
                IdxType,IdxType,ValType,ValType,ValType>())
        ;

    py::class_<Simulation>(m, "Simulation")
        .def(py::init<IdxType, IdxType>())
        .def("append", &Simulation::append)
        .def("upload", &Simulation::upload)
        .def("clear_circuit", &Simulation::clear_circuit)
        .def("run", &Simulation::sim)
        .def("measure", &Simulation::measure)
        .def_static("U3", &Simulation::U3)
        .def_static("U2", &Simulation::U2)
        .def_static("U1", &Simulation::U1)
        .def_static("CX", &Simulation::CX)
        .def_static("ID", &Simulation::ID)
        .def_static("X", &Simulation::X)
        .def_static("Y", &Simulation::Y)
        .def_static("Z", &Simulation::Z)
        .def_static("H", &Simulation::H)
        .def_static("S", &Simulation::S)
        .def_static("SDG", &Simulation::SDG)
        .def_static("T", &Simulation::T)
        .def_static("TDG", &Simulation::TDG)
        .def_static("RX", &Simulation::RX)
        .def_static("RY", &Simulation::RY)
        .def_static("RZ", &Simulation::RZ)
        .def_static("CZ", &Simulation::CZ)
        .def_static("CY", &Simulation::CY)
        .def_static("SWAP", &Simulation::SWAP)
        .def_static("CH", &Simulation::CH)
        .def_static("CCX", &Simulation::CCX)
        .def_static("CSWAP", &Simulation::CSWAP)
        .def_static("CRX", &Simulation::CRX)
        .def_static("CRY", &Simulation::CRY)
        .def_static("CRZ", &Simulation::CRZ)
        .def_static("CU1", &Simulation::CU1)
        .def_static("CU3", &Simulation::CU3)
        .def_static("RXX", &Simulation::RXX)
        .def_static("RZZ", &Simulation::RZZ)
        .def_static("RCCX", &Simulation::RCCX)
        .def_static("RC3X", &Simulation::RC3X)
        .def_static("C3X", &Simulation::C3X)
        .def_static("C3SQRTX", &Simulation::C3SQRTX)
        .def_static("C4X", &Simulation::C4X)
        .def_static("R", &Simulation::R)
        .def_static("SRN", &Simulation::SRN)
        .def_static("W", &Simulation::W)
        .def_static("RYY", &Simulation::RYY)
        ;
}
