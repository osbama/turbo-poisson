mpirun -np 8 python ./precise.py  \
--use_mesh testv4-Eiso-adaptive-alat500-4prec-mesh-final.xml.gz --alat 500 --celldm 1 1 1 \
--chg-parameters 1.0 0.5 0.5 0.5 1.0 --charge-position-units crystal --charge-spread-units bohr \
--epsilon-model constant --epsilon-parameters 4.0 \
--bc_type zero \
--output-prefix "testv4-Eiso-precise" \
--linear_solver "superlu_dist" \
--dump_all \
|tee testv4-Eiso-alat500-precise.log


