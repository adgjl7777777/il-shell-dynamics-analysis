for anion_model in fsi tfsi beti
do
 for T in 298 353 373 423
 do
  export anions=$anion_model
  export Ts=$T
  sbatch -N 1 -t 100000:00:00 -J ${anion_model}_${T} -n 1 -p goldpart2 -o ./out/${anion_model}_${T}.out -e ./out/${anion_model}_${T}.err core.sh
  
 done
done
