#!/bin/sh
echo ${anions} ${Ts} | python3 -u xyz2gro.py > ./out/${anions}_${Ts}_2.out