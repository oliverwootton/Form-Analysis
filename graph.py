import matplotlib.pyplot as plt
from data_read import yData
from data_read import recNo

'Data/u01_movements_wr-su_d_rt_100_20230118-1457.csv'
'Data/u01_movements_wr-su_d_rt_100_20230118-1500.csv'
'Data/u01_movements_wr-su_d_rt_100_20230118-1502.csv'
'Data/u01_movements_wr-su_d_rt_100_20230118-1506.csv'
filename = 'Data/u01_walking_brisk_d_rt_100_20230118-1513.csv'

data = yData(filename)
recNo = recNo(filename)

plt.plot(data, linestyle = 'dotted')
plt.show()
