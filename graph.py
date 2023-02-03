import matplotlib.pyplot as plt
from data_read import *

# filename = 'Data/u01_movements_wr-su_d_rt_100_20230118-1457.csv'
# filename = 'Data/u01_movements_wr-su_d_rt_100_20230118-1500.csv'
filename = 'Data/u01_movements_wr-su_d_rt_100_20230118-1502.csv'
# filename = 'Data/u01_movements_wr-su_d_rt_100_20230118-1506.csv'
# filename = 'Data/u01_walking_brisk_d_rt_100_20230118-1513.csv'

x = Data(filename)

plt.plot(x.yData(), 'r', linestyle = 'dotted')
# plt.plot(xData, 'c', linestyle = 'dotted')
# plt.plot(zData, 'm', linestyle = 'dotted')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.show()
