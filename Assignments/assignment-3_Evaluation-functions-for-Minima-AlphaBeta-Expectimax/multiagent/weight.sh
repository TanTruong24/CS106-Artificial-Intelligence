#!/bin/bash

for ((a=1;a<=1;a++)); do 
	for ((b=-5;b<=0;b++)); do
		for ((c=-5;c<=0;c++)); do 
			for ((d=0;d<=5;d++)); do
				for ((e=-5;e<=0;e++)); do 
					for ((f=-5;f<=0;f++)); do
                        for ((g=-3;g<=3;g++)); do
                            python pacman.py -p ExpectimaxAgent -l smallClassic  -a evalFn=better,depth=3,a=$a,b=$b,c=$c,d=$d,e=$e,f=$f,g=$g --frameTime 0 -q
                            python pacman.py -p ExpectimaxAgent -l smallClassic  -a evalFn=better,depth=3,a=$a,b=$b,c=$c,d=$d,e=$e,f=$f,g=$g --frameTime 0 -q
                            python pacman.py -p ExpectimaxAgent -l smallClassic  -a evalFn=better,depth=3,a=$a,b=$b,c=$c,d=$d,e=$e,f=$f,g=$g --frameTime 0 -q
                            echo $a,$b,$c,$d,$e,$f,$g
                        done    					
					done 
				done
			done 
		done
	done 
done