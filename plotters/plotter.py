import matplotlib.pyplot as plt

xcors = [1, 2, 4]
ycors_static = [848, 430, 221]
ycors_bulk = [840, 420, 220]
ycors_static_lazy = [799, 417, 222]
ycors_bulk_lazy = [471, 284, 260]

# plt.yscale('log')
plt.plot(xcors, ycors_static, label="static")
plt.plot(xcors, ycors_bulk, label="bulk")
plt.plot(xcors, ycors_static_lazy, label="static_lazy")
plt.plot(xcors, ycors_bulk_lazy, label="bulk_lazy")

plt.legend()
plt.savefig("fig.pdf")

xcors_2 = [1, 2, 4, 8]
xcors_2_3 = [1, 2, 4]
xcors_2_2 = [1, 2]

ycors_static_2_250 = [611, 316, 166]
ycors_bulk_2_250 = [597, 308, 164, 83]
ycors_static_lazy_2_250 = [567, 300, 161, 83]
ycors_bulk_lazy_2_250 = [355, 227, 164]
ycors_invasion_2_250 = [614, 325]

ycors_static_2_500 = [679, 346]
ycors_bulk_2_500 = [597, 345, 206]
ycors_static_lazy_2_500 = [615, 323, 199]
ycors_bulk_lazy_2_500 = [406, 240, 194]
ycors_invasion_2_500 = [669, 359]

plt.ylim(ymin=-1)
plt.figure()
plt.plot(xcors_2_3, ycors_static_2_250, label="static")
plt.plot(xcors_2, ycors_bulk_2_250, label="bulk")
plt.plot(xcors_2, ycors_static_lazy_2_250, label="static_lazy")
plt.plot(xcors_2_3, ycors_bulk_lazy_2_250, label="bulk_lazy")
plt.plot(xcors_2_2, ycors_invasion_2_250, label="invasion")

plt.legend()
plt.savefig("fig_2_250.pdf")

plt.ylim(ymin=-1)
plt.figure()
plt.plot(xcors_2_2, ycors_static_2_500, label="static")
plt.plot(xcors_2_3, ycors_bulk_2_500, label="bulk")
plt.plot(xcors_2_3, ycors_static_lazy_2_500, label="static_lazy")
plt.plot(xcors_2_3, ycors_bulk_lazy_2_500, label="bulk_lazy")
plt.plot(xcors_2_2, ycors_invasion_2_500, label="invasion")

plt.legend()
plt.savefig("fig_2_500.pdf")

plt.ylim(ymin=-1)
plt.figure()
plt.plot(xcors_2_3, ycors_static_2_250, label="static")
plt.plot(xcors_2, ycors_bulk_2_250, label="bulk")
plt.plot(xcors_2, ycors_static_lazy_2_250, label="static_lazy")
plt.plot(xcors_2_3, ycors_bulk_lazy_2_250, label="bulk_lazy")
plt.plot(xcors_2_2, ycors_invasion_2_250, label="invasion")
plt.plot(xcors_2_2, ycors_static_2_500, label="static")
plt.plot(xcors_2_3, ycors_bulk_2_500, label="bulk")
plt.plot(xcors_2_3, ycors_static_lazy_2_500, label="static_lazy")
plt.plot(xcors_2_3, ycors_bulk_lazy_2_500, label="bulk_lazy")
plt.plot(xcors_2_2, ycors_invasion_2_500, label="invasion")

plt.legend()
plt.savefig("fig_2.pdf")
