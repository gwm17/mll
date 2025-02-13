import numpy as np
import matplotlib.pyplot as plt

arrays = np.load("losses.npz")
losses = arrays["loss"]
errors = arrays["error"]
epochs = np.arange(len(losses[0]))

fig, ax = plt.subplots(2, 1)
ax[0].plot(epochs, losses[0], label="Training")
ax[0].plot(epochs, losses[1], label="Validation")
ax[0].legend()
ax[1].plot(epochs, errors[:, 0], label="Energy")
ax[1].plot(epochs, errors[:, 1], label="Polar")
ax[1].plot(epochs, errors[:, 2], label="Azim.")
ax[1].plot(epochs, errors[:, 3], label="Vx")
ax[1].plot(epochs, errors[:, 4], label="Vy")
ax[1].plot(epochs, errors[:, 5], label="Vz")
ax[1].legend()
plt.show()
