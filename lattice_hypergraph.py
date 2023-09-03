import matplotlib.pyplot as plt
import numpy as np

n = 4
swap = True

tup = list()
x = list()
y = list()
z = list()
for dimz in range(n ):
    for dimy in range(n  - dimz):
        for dimx in range(n  - dimz):
            if dimy + dimx < n  - dimz:
                tup.append((dimx,dimy,dimz))
                x.append(dimx)
                y.append(dimy)
                z.append(dimz)

if swap:
    for i in range(len(tup)):
        (_x,_y,_z) = tup[i]
        tnew = (n - 1 - _x, n - 1 - _y, _z)
        tup[i] = tnew
        x[i] = tnew[0]
        y[i] = tnew[1]
        z[i] = tnew[2]

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent x pane
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent y pane
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent z pane

ax.scatter(x, y, z, c=z, cmap='viridis', s=50)  # Scatter plot

ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Setting labels
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')

x_ticks = np.arange(0, n , 1)  # Ticks from 0 to 10 with an interval of 1
y_ticks = np.arange(0, n , 1)
z_ticks = np.arange(0, n, 1)

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_zticks(z_ticks)

ax.set_xticklabels([int(tick) for tick in x_ticks])
ax.set_yticklabels([int(tick) for tick in y_ticks])
ax.set_zticklabels([int(tick) for tick in z_ticks])

ax.set_xlim([0-0.1, n+0.1-1])
ax.set_ylim([0-0.1, n+0.1-1])
ax.set_zlim([0-0.1, n+0.1-1])

fig.savefig('plot.pdf', transparent=True)
ax.plot([0, 0], [0, 0], [0, n-1], color=(1, 0.4, 0.4), linewidth=2)
ax.plot([0, 0], [0, n-1], [0, 0], color=(1, 0.4, 0.4), linewidth=2)
ax.plot([0, n-1], [0, 0], [0, 0], color=(1, 0.4, 0.4), linewidth=2)

plt.show()
fig.savefig('plot_w_he.pdf', transparent=True)