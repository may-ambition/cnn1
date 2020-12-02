import matplotlib.pyplot as plt

a = [1, 2, 3, 4] # y 是 a的值，x是各个元素的索引
b = [5, 6, 7, 8]

plt.plot(a, b, 'r--', label = 'aa')
plt.xlabel('this is x')
plt.ylabel('this is y')
plt.title('this is a demo')
plt.legend() # 将样例显示出来

plt.plot()
plt.show()

