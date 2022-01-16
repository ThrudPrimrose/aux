import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

"""
x_cors = [1, 2, 4, 6]
titles = ["1", "2", "4", "6"]

y_static = [0, 362.698, 183.176, 114.566]
y_lazy = [0, 310.655, 153.401, 109.859]

y_bulk = [618.7, 308.3, 161.352, 110.179]
y_steal = [684.051, 351.557, 184.843, 0.0]

y_lazybulk = [465.315, 0, 186.725, 162.848]
y_lazysteal = [345.292, 0, 0, 0]

y_invasionbulk = [0, 330.404, 182.541, 0]
y_invasionsteal = [0, 0, 184.748, 0]
"""

x_cors = [1, 2]
titles = ["1", "2"]

y_static = [683.734, 431.01]
y_lazy = [550.045, 335.259]

y_bulk = [612.304, 319.637]
y_steal = [685.579, 352]

y_lazybulk = [428.62, 264.565]
y_lazysteal = [352.072, 189.358]

y_invasionbulk = [620.69, 333.505]
y_invasionsteal = [691.512, 0]


def plot_lines(name, dataset, labels, s, title):
    with matplotlib.backends.backend_pdf.PdfPages(name) as pdf:
        plt.figure()
        plt.title(title)
        for i in range(s):
            #title = titles[i]
            y_cors = dataset[i]
            # ymin = min(ys[i])
            # ymax = max(ys[i])
            label = labels[i]
            plt.plot(x_cors, y_cors, label=label)
        plt.legend()
        pdf.savefig()


labels_exp = ["static-0", "lazy-0", "bulk-1", "steal-2",
              "lazy-bulk-1", "lazy-steal-2", "invasion-bulk-1", "invasion-steal-2"]
_y = [y_static, y_lazy, y_bulk, y_steal, y_lazybulk,
      y_lazysteal, y_invasionbulk, y_invasionsteal]

plot_lines("pond-1.pdf", _y, labels_exp, 8,
           "runs of pond")
