#include <cassert>
#include <iostream>

#include "CoreTypes.hpp"
#include "IQuantumApi.hpp"
#include "ITranslator.hpp"
#include "QPFactory.hpp"

extern "C" Result ResultOne = nullptr;
extern "C" Result ResultZero = nullptr;

//extern "C" double Sample_VQE_EstimateTermExpectation(int64_t state1__count, int64_t* state1, int64_t state2__count, int64_t* state2, double phase, int64_t ops_count, enum PauliId* ops, double coeff, int64_t nSamples);
extern "C" double Microsoft__Quantum__Samples__Chemistry__SimpleVQE__GetEnergyHydrogenVQE__body();

using namespace quantum;
using namespace std;

extern "C" void print_double(double x)
{
    std::cout << "double value is " << x << std::endl;
}

extern "C" void print_bool(bool x)
{
    std::cout << "bool value is " << x << std::endl;
}

extern "C" void print_result(Result* r)
{
    std::cout << "Result value is " << r << std::endl;
}



//extern "C" double __quantum__qis__intAsDouble(int64_t x)
//{
//return static_cast<double>(x);
//}

extern "C" Result* measure(QUBIT* measureops, QUBIT* registers)
{
    std::cout << "Here has some issues in qis__mesure()" << std::endl;
    Result* res = new Result[1000];
    //Result res[4] = {Result(0), Result(0), Result(0), Result(0)};
    return res;
}

int main()
{
    //const int64_t n = 1;
    //int64_t states1[n] = {(int64_t)0};
    //int64_t states2[n] = {(int64_t)0};
    //double phase = 3.1415927/4;
    //double coeff = 0.17;
    //int64_t nSamples = 1;
    //const int64_t ops_n = 1;
    //enum PauliId ops[ops_n] = {PauliId_Z};

    std::cout << "*** Testing QIR VQE example with DM-Sim ***" << std::endl;
    double jwTermEnergy = 0;
    //jwTermEnergy = Sample_VQE_EstimateTermExpectation(
    //n, states1, n, states2, phase, ops_n, ops, coeff, nSamples);
    jwTermEnergy = Microsoft__Quantum__Samples__Chemistry__SimpleVQE__GetEnergyHydrogenVQE__body();
    
    
    std::cout << "jwTermEnergy is " << jwTermEnergy << std::endl;

    return 0;
}
