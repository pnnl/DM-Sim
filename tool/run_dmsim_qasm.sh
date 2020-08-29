python dmsim_qasm.py -i vqe_uccsd_n8.qasm -o vqe_uccsd_n8.py
cp ../src/dmsim_py_omp_wrapper.so .
python vqe_uccsd_n8.py 8 2
