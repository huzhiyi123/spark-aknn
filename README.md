# spark-aknn
spark-aknn

. \
├── README.md \
├── aknn-main.py \
├── datasets.py      read data function \
└── sift_small_5MB.tar.gz data
## 设置token url的方法
### file .git/config 
```
[remote "origin"]
	url = https://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@github.com/huzhiyi123/spark-aknn.git \
	fetch = +refs/heads/*:refs/remotes/origin/*
```
1. 先删除远程url
2. git remote add origin https://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@github.com/huzhiyi123/spark-aknn.git
3. git push set upstream
   