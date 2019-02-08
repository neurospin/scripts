#!/bin/sh

#Originally written by Saad Jbabdi
# University of Oxford, FMRIB Centre
# and posted on JISCMail FSL Archives Feb 3 2012
# modified by Neda Jahanshad (USC/ENIGMA-DTI)
# to take in vertically oriented bvec files

if [ "$3" == "" ] ; then 
 echo "Usage: <original bvecs> <rotated bvecs> <ecclog>"
 echo ""
 echo "<ecclog>	is the output log file from ecc"
 echo ""
 exit 1;
fi

i=$1
o=$2
ecclog=$3

if [ ! -e $1 ] ; then
	echo "Source bvecs $1 does not exist!"
	exit 1
fi
if [ ! -e $ecclog ]; then
	echo "Ecc log file $3 does not exist!"
	exit 1
fi

nline=$(cat $i | wc -l )
if [ $nline -gt 3 ]
then
echo "the file is vertical and will be transposed"
awk '
{
for (k=1; k<=NF; k++)  {
a[NR,k] = $k
}
}
NF>p { p = NF }
END {
for(j=1; j<=p; j++) {
str=a[1,j]
for(k=2; k<=NR; k++){
str=str" "a[k,j];
}
print str
}
}' $i > ${i}_horizontal

i=${i}_horizontal
fi



ii=1
rm -f $o
tmpo=${o}$$
cat ${ecclog} | while read line; do
    echo $ii
    if [ "$line" == "" ];then break;fi
    read line;
    read line;
    read line;

    echo $line  > $tmpo
    read line    
    echo $line >> $tmpo
    read line    
    echo $line >> $tmpo
    read line    
    echo $line >> $tmpo
    read line   
    
    m11=`avscale $tmpo | grep Rotation -A 1 | tail -n 1| awk '{print $1}'`
    m12=`avscale $tmpo | grep Rotation -A 1 | tail -n 1| awk '{print $2}'`
    m13=`avscale $tmpo | grep Rotation -A 1 | tail -n 1| awk '{print $3}'`
    m21=`avscale $tmpo | grep Rotation -A 2 | tail -n 1| awk '{print $1}'`
    m22=`avscale $tmpo | grep Rotation -A 2 | tail -n 1| awk '{print $2}'`
    m23=`avscale $tmpo | grep Rotation -A 2 | tail -n 1| awk '{print $3}'`
    m31=`avscale $tmpo | grep Rotation -A 3 | tail -n 1| awk '{print $1}'`
    m32=`avscale $tmpo | grep Rotation -A 3 | tail -n 1| awk '{print $2}'`
    m33=`avscale $tmpo | grep Rotation -A 3 | tail -n 1| awk '{print $3}'`

    X=`cat $i | awk -v x=$ii '{print $x}' | head -n 1 | tail -n 1 | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}' `
    Y=`cat $i | awk -v x=$ii '{print $x}' | head -n 2 | tail -n 1 | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}' `
    Z=`cat $i | awk -v x=$ii '{print $x}' | head -n 3 | tail -n 1 | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}' `
    rX=`echo "scale=7;  ($m11 * $X) + ($m12 * $Y) + ($m13 * $Z)" | bc -l`
    rY=`echo "scale=7;  ($m21 * $X) + ($m22 * $Y) + ($m23 * $Z)" | bc -l`
    rZ=`echo "scale=7;  ($m31 * $X) + ($m32 * $Y) + ($m33 * $Z)" | bc -l`

    if [ "$ii" -eq 1 ];then
	echo $rX > $o;echo $rY >> $o;echo $rZ >> $o
    else
	cp $o $tmpo
	(echo $rX;echo $rY;echo $rZ) | paste $tmpo - > $o
    fi
    
    let "ii+=1"

done
rm -f $tmpo

