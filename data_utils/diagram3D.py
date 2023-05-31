import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits import mplot3d
import matplotlib.colors as colors

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')    
matrix = np.array([[1.76972277e+02, 1.63289191e+02, 1.01219641e+01, 2.41330293e+01,
        2.30300844e+01, 3.98533667e+01, 2.46068202e-01, 1.15366535e+01,
        1.63441166e+00, 4.91829539e+01],
       [1.24814623e+02, 3.06037074e+01, 3.63168992e+01, 1.11988664e+02,
        8.21235557e+01, 2.02379701e+01, 1.68274121e+01, 6.95491962e+01,
        2.29472274e+00, 5.24325005e+00],
       [7.28155805e+00, 3.55339916e+01, 9.93966892e+01, 1.30890352e+01,
        1.21043431e+02, 7.24095469e+01, 7.36463434e+01, 6.82793395e+00,
        6.01593154e+01, 1.06121556e+01],
       [2.46782178e-01, 2.18420951e+01, 2.11530250e+01, 2.46652854e+00,
        1.52048165e+02, 4.39071856e+01, 4.31671849e+01, 7.71855291e+01,
        1.37316920e+02, 6.66584563e-01],
       [4.27142792e-02, 1.28951495e+02, 4.31797553e+01, 4.69117999e+00,
        1.20023356e+02, 1.51012191e+01, 5.01272444e+01, 3.48061248e+01,
        2.01611071e+01, 8.29158042e+01],
       [1.69251715e+01, 3.66048062e+01, 3.46047398e+01, 1.82253013e+02,
        3.34131703e+01, 3.65836587e+00, 6.29588567e+01, 6.51218750e+01,
        3.87513825e+00, 6.05848633e+01],
       [9.90933179e+01, 6.71795917e+01, 1.39381856e+01, 1.19971199e+02,
        5.49510156e+01, 3.58504089e+01, 6.16977888e+01, 2.45269742e+01,
        2.18246821e+01, 9.66836094e-01],
       [1.75106540e+00, 1.70163277e+01, 6.37105828e+01, 9.95030095e+01,
        1.40007304e+01, 7.45541250e+00, 1.15808413e+01, 1.31915834e+02,
        8.45451562e+01, 6.85210402e+01],
       [5.73634114e+01, 1.17639748e+02, 2.29492140e+01, 2.96026211e+01,
        6.25851658e+01, 2.58413074e+01, 1.66223136e+01, 3.13456981e+01,
        3.46165366e+01, 1.01433984e+02],
       [2.78029235e+00, 7.76412524e+01, 6.15104025e+00, 2.11425763e+01,
        1.32793102e+02, 2.58338283e+00, 1.16817258e+02, 5.53668839e+01,
        3.83750916e+01, 4.63491204e+01],
       [6.68155937e+00, 1.73955780e+02, 7.33127330e+01, 1.59705214e+01,
        1.57638391e+01, 1.06159975e+01, 6.65405342e+01, 2.96410127e+01,
        4.32612553e+01, 6.42567674e+01],
       [3.71838130e+00, 4.78300737e+01, 4.62087006e+01, 1.75552694e+01,
        1.19882993e+01, 8.88185576e+00, 1.32280791e+02, 1.67428660e+02,
        4.37778798e+01, 2.03300895e+01],
       [1.50543782e+01, 2.92615254e+01, 1.11681829e+01, 4.84809300e+01,
        3.73950127e+01, 6.52326192e+00, 1.68982707e+00, 8.31068261e+01,
        1.42361547e+02, 1.24958508e+02],
       [3.23737392e+01, 1.35709203e+01, 3.97086059e-01, 5.86794613e+00,
        4.79887849e+01, 1.68561812e+00, 3.83831959e+01, 8.01655262e+01,
        2.48111321e+02, 3.14558626e+01],
       [1.87332858e+01, 6.29647347e+01, 4.60064197e+00, 3.83656881e+00,
        6.27757627e+01, 3.41531825e+01, 1.32705525e+01, 2.33369051e+02,
        5.31662464e+01, 1.31299738e+01],
       [3.08202743e+01, 1.37984120e+02, 4.13229295e+01, 6.35067059e+01,
        2.44388216e+01, 8.81327205e+01, 4.80391787e+00, 2.89944903e+01,
        7.16075109e+01, 8.38850893e+00],
       [1.04587801e+02, 3.95798568e+01, 8.92122996e+01, 7.72246456e+01,
        8.23894229e+00, 1.78897030e+00, 8.40463373e+00, 1.25836695e+01,
        1.55381696e+02, 2.99748503e+00],
       [1.72180160e+00, 3.98013827e+01, 4.42122940e+01, 2.43860258e+01,
        1.02130461e+02, 1.09607394e+02, 2.93034325e+01, 9.17782848e+01,
        1.51237482e+01, 4.19351752e+01],
       [2.51730301e+01, 8.00936590e+01, 2.11346638e+02, 1.81451240e+01,
        3.35862444e+00, 1.29932489e+02, 1.71375930e+01, 3.79543945e+00,
        2.10938682e-01, 1.08064639e+01],
       [3.62433899e+01, 3.23895485e+01, 3.72293613e+01, 1.91018402e+01,
        8.22110566e+00, 2.10707903e+00, 1.90415856e+02, 1.60619214e+02,
        7.84952348e+00, 5.82308244e+00]])

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
# plt.savefig('/imgs/3dDiagram_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))