for anion_model in fsi tfsi beti
do
  export anions=$anion_model
  sbatch -N 1 -t 100000:00:00 -J ${anion_model} -n 1 -p goldpart2 -o ./out/${anion_model}.out -e ./out/${anion_model}.err core.sh
done
