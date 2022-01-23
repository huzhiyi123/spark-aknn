#!bash
#bash 2.sh 19.log
#bash 5.sh 20.log
#nohup bash /runall.sh &>> cur4.log&
#bash 2.sh 1.log
log=2.log
aknnfold=/aknn
logfold=/aknn/alllog/1/
cd $aknnfold
bash 5.sh $1
#cat test/params.py >> $logfold$log
