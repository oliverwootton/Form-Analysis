import matplotlib.pyplot as plt
from data_read import *

filename1 = 'Data/u01_movements_wr-su_d_rt_100_20230118-1457.csv'
filename2 = 'Data/u01_movements_wr-su_d_rt_100_20230118-1500.csv'
filename3 = 'Data/u01_movements_wr-su_d_rt_100_20230118-1502.csv'
filename4 = 'Data/u01_movements_wr-su_d_rt_100_20230118-1506.csv'
filename5 = 'Data/u01_walking_brisk_d_rt_100_20230118-1513.csv'

x1 = Data(filename1)
x2 = Data(filename2)
x3 = Data(filename3)
x4 = Data(filename4)
x5 = Data(filename5)

# plt.plot(x1.yData()[1500:3000:2], 'r', linestyle = 'dotted')
# plt.plot(x2.yData()[1500:3000:2], 'g', linestyle = 'dotted')
# plt.plot(x3.yData()[1500:3000:2], 'b', linestyle = 'dotted')
plt.plot(x4.yData(), 'm', linestyle = 'dotted')
# plt.plot(x5.yData()[1500:3000:2], 'c', linestyle = 'dotted')
# plt.plot(xData, 'c', linestyle = 'dotted')
# plt.plot(zData, 'm', linestyle = 'dotted')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.show()
