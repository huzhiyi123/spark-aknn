#!bash
#bash 2.sh 19.log
#bash 5.sh 20.log
#nohup bash /runall.sh &>> cur4.log&
#bash 2.sh 1.log
log=16.log
logfold=/workspace/alllog/1/1/
bash 5.sh $log
#cat test/params.py >> $logfold$log
