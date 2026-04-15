for anion_model in tfsi
do
 for T in 298 353 373 423
 do
  echo ${anion_model} ${T} | nohup python3 -u com.py > ./${T}K.out &
 done
done
