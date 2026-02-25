mpirun -np 1 python ./stdalone-pbc-v1.py  \
--alat 20 --mesh 8 8 16 --celldm 1 1 2 \
--chg-parameters 1.0 10.0 10.0 6.0 1.0 --charge-position-units bohr \
--epsilon-model top-bottom --epsilon-parameters 1.0 4.0 0.5 0.0001 --epsilon-position-units crystal \
--adaptive fenics --adaptive_tol 0.001 --linear_solver 'superlu_dist' \
--file-output-level 0 --interactive\
 |tee test.log 
