tolerance=0.001

for pos in `seq 1400 10 1600`
do
   mpirun -np 1 python ./stdalone-v4.py  \
   --alat 1500 --mesh 20 20 40 \
   --celldm 1 1 2 --chg-parameters 1.0 750 750 $pos 4.0\
   --charge-position-units bohr --charge-spread-units bohr \
   --epsilon-model top-bottom\
   --epsilon-parameters 1.0 4.0 0.5 0.0001 --epsilon-position-units crystal\
   --adaptive fenics --adaptive_tol $tolerance\
   --bc_type zero \
   --dump_mesh \
   --output-prefix "Einter-adaptive-pos$pos" \
   --duallinear_solver "superlu_dist" \
   --linear_solver "superlu_dist" 

   mpirun -np 1 python ./precise.py \
   --alat 1500 --use_mesh Einter-adaptive-pos$pos-mesh-final.xml.gz \
   --celldm 1 1 2 --chg-parameters 1.0 750 750 $pos 4.0\
   --charge-position-units bohr --charge-spread-units bohr \
   --epsilon-model top-bottom\
   --epsilon-parameters 1.0 4.0 0.5 0.0001 --epsilon-position-units crystal\
   --bc_type zero \
   --dump_all \
   --output-prefix "Einter-precise-alat$alat-pos$pos" \
   --linear_solver superlu_dist --preconditioner ilu

done

