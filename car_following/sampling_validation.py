import numpy as np 
import random

intensity_abs_max = 3
intensity_abs_min = 1

# Make sure that intensity is always outside the rejection range.
intensity = random.uniform(-intensity_abs_max, intensity_abs_max)
while intensity > -intensity_abs_min and intensity < intensity_abs_min:
    intensity = random.uniform(-intensity_abs_max, intensity_abs_max)

print("Intensity:", intensity)

durations = np.linspace(0.1, 2.5, 20)
print("Durations:", durations)

abs_intensity = abs(intensity)
print("Abs intensity:", abs_intensity)

intensity_bucket = np.linspace(intensity_abs_min, intensity_abs_max,len(durations))
print("Intensity bucket:", intensity_bucket)

loc = np.searchsorted(intensity_bucket, abs_intensity)
print("Loc:", loc)

left = loc 
right = len(durations) - loc
print("Left:", left, "Right:", right)

probabilities_left = np.linspace(0.0, 10, left)
print("Probabilities left:", probabilities_left, probabilities_left.sum())

probabilities_right = np.linspace(10, 0.0, right)
print("Probabilities right:", probabilities_right, probabilities_right.sum())

probabilities = np.concatenate((probabilities_left, probabilities_right))
probabilities /= probabilities.sum()

duration = round(np.random.choice(durations, 1, p=probabilities)[0], 1)
print("Duration:", duration)

# Plot the probabilities
import matplotlib.pyplot as plt
plt.plot(probabilities)
plt.show()


