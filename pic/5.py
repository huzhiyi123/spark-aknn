import matplotlib.pyplot as plt
import numpy as np

t1 = [55031.230211257935,279392.8453922272,1339.5884037017822]
t2 = [ 22974.760055541992 , 22897.218227386475 , 77.54182815551758]


plt.pie(t1,
        labels=['A','B','C'], # 设置饼图标签
        colors=["#d5695d", "#5d8ca8", "#65a479" ], # 设置饼图颜色
        #explode=(0, 0.2, 0, 0), # 第二部分突出显示，值越大，距离中心越远
        explode=(0,0, 0.6), # 第二部分突出显示，值越大，距离中心越远
        autopct='%.2f%%' 
       )
plt.title("RUNOOB Pie Test")
plt.savefig("pie.png")