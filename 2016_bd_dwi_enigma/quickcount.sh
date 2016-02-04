for i in `seq 10 99`; do 
	
	m=0
	n=0
	for dtifile in dti/*.nii; do 
		if [[ $dtifile =~  $i ]]; then 
			m=$((m+1)) 
		fi
	done
	for t1file in t1/*.nii; do 
		if [[ $t1file =~  Ed$i ]]; then 
			n=$((n+1)) 
		fi
		
	done
	if ! [ "$m" == "$n" ]
		echo "For the ID $i, we have $m nii files in dti and $n in t1:" 
	fi
done
