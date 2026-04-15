#!/bin/bash
##SBATCH -J rdf
##SBATCH -p intelpart
##SBATCH -o %x.o.%j.out
##SBATCH -e %x.e.%j.err
##SBATCH -N 1
##SBATCH -n 1
###SBATCH --nodelist=node08
##SBATCH -t 100000:00:00



python3 -u rdf.py ${anions} ${Ts}
