##Created by :  
## Creation Date: 
## Description: Script is for make and run of hipCopyHammer_D2D test
## Modifications: 

#!/bin/sh

current=`pwd`

if [ -e "$current/hipCopyHammer_D2D" ] ; then
	make clean
	make > hipCopyHammer_D2D_build.log 2>&1
else
	make > hipCopyHammer_D2D_build.log 2>&1
fi

if [ -e "$current/hipCopyHammer_D2D" ] ; then
   echo "[STEPS]" > Results.ini
   echo "Number=2" >> Results.ini
   echo " " >> Results.ini
   echo "[STEP_001]" >> Results.ini
   echo "Description=hipCopyHammer_D2D built successfully" >> Results.ini
   echo "Status=Passed" >> Results.ini
else
   echo "[STEPS]" > Results.ini
   echo "Number=1" >> Results.ini
   echo " " >> Results.ini
   echo "[STEP_001]" >> Results.ini
   echo "Description=hipCopyHammer_D2D building Failed" >> Results.ini
   echo "Status=Failed" >> Results.ini
   exit 1
fi


./hipCopyHammer_D2D > hipCopyHammer_D2D_run.log 2>&1


RUN_PATT1='Aborted'
RUN_PATT2='Segmentation'
RUN_PATT3='FAILED'


RUN_LOG='hipCopyHammer_D2D_run.log'

if [ -s $RUN_LOG ] ; then
 if grep -qiE "$RUN_PATT1|$RUN_PATT2|$RUN_PATT3" $RUN_LOG;
then
   echo " " >> Results.ini
   echo "[STEP_002]" >> Results.ini
   echo "Description=hipCopyHammer_D2D Failed" >> Results.ini
   echo "Status=Failed" >> Results.ini
 else
   echo " " >> Results.ini
   echo "[STEP_002]" >> Results.ini
   echo "Description=hipCopyHammer_D2D passed" >> Results.ini
   echo "Status=Passed" >> Results.ini
   
 fi
fi

