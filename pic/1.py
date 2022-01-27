# 导入相关数据包
import matplotlib.pyplot as plt
import pandas as pd

a=[0.0923,0.7279,
0.04139999999999999,0.6626,
0.023099999999999996,0.6022,
0.021599999999999994,0.6059,
0.023999999999999994,0.5922,
0.010999999999999996,0.5659,
0.026599999999999985,0.6079,
0.018299999999999997,0.5698,
0.015099999999999988,0.5791,
0.020599999999999993,0.5884]
x=range(10)
y1=[]
y2=[]
print((len(a)/2))
for i in range(int(len(a)/2)):
    y1.append(a[i*2])
    y2.append(a[i*2+1])


for i in range(len(y1)):
    y1[i] = (0.1+y1[i])/0.1
for i in range(len(y2)):
    y2[i] = (1-y2[i])

print(y1)
print(y2)
"""
[1.9229999999999998, 1.414, 1.2309999999999999, 1.216, 1.24, 1.1099999999999999, 1.2659999999999998, 1.183, 1.1509999999999998, 1.206]
[0.7279, 0.6626, 0.6022, 0.6059, 0.5922, 0.5659, 0.6079, 0.5698, 0.5791, 0.5884]
"""


fig = plt.figure(figsize=(10,8),dpi=80)
# 添加Axes坐标轴实例，创建1个画板
ax = fig.add_subplot(111)  





# 制作第一条折现
lin1 = ax.plot(x, y2, label='cost2', color='g')
ax.set_xlabel('partition rate')
# 设置Y轴1
ax.set_ylabel('cost2')


labs = [label.get_label() for label in lin1]
ax.legend(lin1,labs)
# 增加网格线
#ax.grid()


plt.ylim(0.2,0.5)
plt.savefig("4.png")


"""
fig = plt.figure(figsize=(10,8),dpi=80)
# 添加Axes坐标轴实例，创建1个画板
ax = fig.add_subplot(111)  
# 制作第一条折现
lin1 = ax.plot(x, y1, label='xx', color='r')
ax.set_xlabel('xxx')
# 设置Y轴1
ax.set_ylabel('xx')
# 使用twinx()函数实现共用一个x轴
ax2 = ax.twinx()
# 制作第二条折现
lin2 = ax2.plot(x, y2, label='xx', color='green')
# 设置Y轴2
ax2.set_ylabel('xxx')
# 合并图例
lines = lin1+lin2
labs = [label.get_label() for label in lines]
ax.legend(lines,labs)
# 增加网格线
ax.grid()




plt.show()
plt.savefig("1.png")
"""