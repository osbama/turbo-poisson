for alat in 500
do
mpirun -np 1 python ./stdalone-v4.py  \
--alat $alat --mesh 4 4 4 --celldm 1 1 1 \
--chg-parameters 1.0 0.5 0.5 0.5 1.0 --charge-position-units crystal --charge-spread-units bohr \
--epsilon-model constant --epsilon-parameters 4.0 \
--adaptive fenics --adaptive_tol 0.0005 \
--bc_type zero \
--output-prefix "testv4-Eiso-adaptive-alat$alat-4prec" \
--duallinear_solver "superlu_dist" \
--linear_solver "superlu_dist" \
|tee testv4-Eiso-adaptive-alat$alat4prec.log
done

