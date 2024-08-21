import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = 'a'
t = 't'
f = 'f'

points = [
            #1
            {f:4, a: 0.3, t: .11},
            {f:4, a: 0.4, t: .20},
            {f:4, a: 0.5, t: .36},
            {f:4, a: 0.6, t: .53},
            {f:4, a: 0.7, t: .67},
            {f:4, a: 0.8, t: .80},
            {f:4, a: 0.9, t: .90},
            {f:4, a: 1.0, t: 1.0},

            #2
            {f:8, a: 0.3, t: .06},
            {f:8, a: 0.4, t: .14},
            {f:8, a: 0.5, t: .24},
            {f:8, a: 0.6, t: .36},
            {f:8, a: 0.7, t: .50},
            {f:8, a: 0.8, t: .61},
            {f:8, a: 0.9, t: .80},
            {f:8, a: 1.0, t: .90},

            #3
            {f:12, a: 0.3, t: .03},
            {f:12, a: 0.4, t: .08},
            {f:12, a: 0.5, t: .14},
            {f:12, a: 0.6, t: .24},
            {f:12, a: 0.7, t: .33},
            {f:12, a: 0.8, t: .47},
            {f:12, a: 0.9, t: .57},
            {f:12, a: 1.0, t: .67},

            #4
            {f:16, a: 0.3, t: .00},
            {f:16, a: 0.4, t: .04},
            {f:16, a: 0.5, t: .10},
            {f:16, a: 0.6, t: .15},
            {f:16, a: 0.7, t: .22},
            {f:16, a: 0.8, t: .33},
            {f:16, a: 0.9, t: .40},
            {f:16, a: 1.0, t: .50},

            #5
            {f:20, a: 0.4, t: .00},
            {f:20, a: 0.5, t: .06},
            {f:20, a: 0.6, t: .12},
            {f:20, a: 0.7, t: .17},
            {f:20, a: 0.8, t: .24},
            {f:20, a: 0.9, t: .32},
            {f:20, a: 1.0, t: .40},
            
            #6
            {f:24, a: 0.4, t: .00},
            {f:24, a: 0.5, t: .03},
            {f:24, a: 0.6, t: .07},
            {f:24, a: 0.7, t: .12},
            {f:24, a: 0.8, t: .17},
            {f:24, a: 0.9, t: .34},
            {f:24, a: 1.0, t: .21},

            #7
            {f:28, a: 0.4, t: .00},
            {f:28, a: 0.5, t: .02},
            {f:28, a: 0.6, t: .05},
            {f:28, a: 0.7, t: .09},
            {f:28, a: 0.8, t: .13},
            {f:28, a: 0.9, t: .19},
            {f:28, a: 1.0, t: .24},
            
            #8
            {f:34, a: 0.5, t: .00},
            {f:34, a: 0.6, t: .03},
            {f:34, a: 0.7, t: .06},
            {f:34, a: 0.8, t: .10},
            {f:34, a: 0.9, t: .14},
            {f:34, a: 1.0, t: .18},

            #9
            {f:36, a: 0.5, t: .00},
            {f:36, a: 0.6, t: .02},
            {f:36, a: 0.7, t: .04},
            {f:36, a: 0.8, t: .07},
            {f:36, a: 0.9, t: .10},
            {f:36, a: 1.0, t: .12},
        ]

amplitudeFract  = np.array([])
torque          = np.array([])
freqHz          = np.array([])

for point in points:
    amplitudeFract = np.append(amplitudeFract, point[a])
    torque = np.append(torque, point[t])
    freqHz = np.append(freqHz, point[f])

torqueXfreq = torque * freqHz
X = np.column_stack((torque, freqHz, torqueXfreq))
X = sm.add_constant(X)

model = sm.OLS(amplitudeFract, X)
results = model.fit()

# print(results.summary())

coefficients = results.params
alpha = coefficients[0]
beta = coefficients[1]
gamma = coefficients[2]
delta = coefficients[3]


# Generate a grid of torque and freq values
torqueVals = np.linspace(min(torque), max(torque), 30)
freqHzVals = np.linspace(min(freqHz), max(freqHz), 30)
torqueGrid, freqHzGrid = np.meshgrid(torqueVals, freqHzVals)

# Calculate predicted a values
amplitudeFractPred = alpha + beta * torqueGrid + gamma * freqHzGrid + delta * torqueGrid * freqHzGrid

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting original data points
ax.scatter(torque, freqHz, amplitudeFract, color='red', label='Data Points')

# Plotting the fitted model surface
ax.plot_surface(torqueGrid, freqHzGrid, amplitudeFractPred, color='blue', alpha=0.5, label='Fitted Model')

ax.set_xlabel('torque')
ax.set_ylabel('freqHz')
ax.set_zlabel('ampligudeFract')
ax.set_title('3D Visualization of Model Fit')
plt.legend()

plt.show()



print('amplitudeFract = alpha + beta * torque + gamma * freqHz + delta * torque * freqHz')
print(f'alpha = {alpha}')
print(f'beta = {beta}') 
print(f'gamma = {gamma}')
print(f'delta = {delta}')