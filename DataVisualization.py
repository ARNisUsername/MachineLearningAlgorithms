#---------MATPLOTLIB---------

#Subplots
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=2)

x = [1,2,3]
y = [1,2,3]
ax[0].plot(x, y, color='r', label='0th axis')
ax[1].plot(x, y, color='g', label='1st axis')
ax[0].set_xlim(1,4)
ax[1].set_xlim(1,4)

fig.legend()
fig.tight_layout()
plt.show()

#Two plot in same plot
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.9,0.9]) #[how much away from left, how amount away from bottom, length, width]
ax2 = fig.add_axes([0.3,0.5,0.3,0.3])
ax1.plot([1,2,3],[1,2,3])
ax2.plot([1,2,3],[1,2,3])
plt.show()

#-------------SEABORN------------

import seaborn as sns
tips = sns.load_dataset('tips')
