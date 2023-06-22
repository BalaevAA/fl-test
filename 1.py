import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits import mplot3d
import matplotlib.colors as colors

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')    
matrix = np.array([[ 350,  422,  338,  364,  476,    6,  207,   24,   50,  254],
       [ 450,  794,  434,  714,  623,  393,  210,   80,  105,  378],
       [ 462, 1158,  604,  813, 1078,  882,  476,  210,  397,  541],
       [ 906, 1471, 1013, 1087, 1286, 1254,  661,  248,  785,  651],
       [ 946, 1714, 1090, 1501, 1382, 1389,  942,  699, 1211,  860],
       [1143, 1722, 1240, 1678, 1829, 1782, 1328,  909, 1599, 1092],
       [1234, 2164, 1599, 2014, 1957, 1997, 1336, 1026, 1855, 1238],
       [1485, 2413, 1651, 2237, 2442, 2366, 1696, 1473, 2336, 1337],
       [1842, 2494, 2084, 2549, 2922, 2628, 1714, 1836, 2338, 1362],
       [2338, 2600, 2558, 2775, 3421, 2952, 2065, 2051, 2422, 1421]])

len_x, len_y = matrix.shape
_x = np.arange(0,len_x,1)
_y = np.arange(0,len_y,1)

xpos, ypos = np.meshgrid(_x, _y)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

dx = np.ones_like(zpos)*0.5
dy = dx.copy()*0.5
dz = matrix.flatten()

# cmap=plt.cm.magma(plt.Normalize(0,100)(dz))
cmap = plt.cm.get_cmap('jet') # Get desired colormap
max_height = np.max(dz)   # get range of colorbars
min_height = np.min(dz)

# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz]

ax.bar3d(xpos+0.32, ypos-0.3, zpos, dx, dy, dz, color=rgba, edgecolor='k')

ax.set_xlabel('class')
# ax.set_xticks(np.arange(len_x+1))
# ax.set_xticklabels(['1000','500','100','50','0'])
# ax.set_xlim(0,4)
ax.set_ylabel('client')
# ax.set_yticks(np.arange(len_y+1))
# ax.set_yticklabels(['0.5','1.','1.5','2.','2.5','3.','3.5','4.','4.5','5.'])
# ax.set_ylim(-0.5,10)
ax.set_zlabel('number od data on')
# ax.set_zlim(0,100)
# ax.view_init(ax.elev, ax.azim+10)
plt.show()