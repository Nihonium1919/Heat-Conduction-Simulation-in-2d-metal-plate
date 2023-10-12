# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:13:15 2023

@author: Niharika
"""

import numpy as np
import matplotlib.pyplot as plt

# Define constants
plate_size = 0.6  # meters
num_points = 50  # Number of grid points in each dimension
thermal_conductivity = 205  # W/m-K
density = 7850  # kg/m^3
specific_heat_capacity = 450  # J/kg-K
time_step = 0.01  # Time step (s)
total_time = 100 # Total simulation time (s)

# Calculate grid spacing
dx = plate_size / (num_points - 1)
dy = plate_size / (num_points - 1)

# Initialize temperature distribution matrix
T = np.zeros((num_points, num_points))
T_new = np.copy(T)

# Set boundary conditions (e.g., keeping the edges at a constant temperature)
T[:, 0] = 400.0  # Left boundary
T[:, -1] = 600.0  # Right boundary
T[0, :] = 800.0  # Top boundary
T[-1, :] = 900.0  # Bottom boundary

# Define a function to solve the heat equation using implicit finite difference method
def solve_heat_equation(T, T_new, num_points, dx, dy, thermal_conductivity, density, specific_heat_capacity, time_step):
    alpha = thermal_conductivity / (density * specific_heat_capacity)
    r_x = alpha * time_step / dx**2
    r_y = alpha * time_step / dy**2

    for i in range(1, num_points - 1):
        for j in range(1, num_points - 1):
            T_new[i, j] = T[i, j] + r_x * (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) + r_y * (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1])

    T_new, T = T, T_new

    return T

# Initialize time and iteration variables
time = 0.0
iterations = int(total_time / time_step)

# Create a list to store temperature distributions at different times
temperature_history = [np.copy(T)]

# Solve the heat equation over time
for _ in range(iterations):
    T = solve_heat_equation(T, T_new, num_points, dx, dy, thermal_conductivity, density, specific_heat_capacity, time_step)
    temperature_history.append(np.copy(T))
    time += time_step

# Calculate heat flux using forward divided difference method
heat_flux_x = -thermal_conductivity * np.gradient(T, axis=1) / dx
heat_flux_y = -thermal_conductivity * np.gradient(T, axis=0) / dy 

# Visualization of temperature distribution and heat flux
plt.figure(figsize=(12, 6))

# Plot temperature distribution at a specific time (e.g., the final time)
plt.subplot(1, 2, 1)
plt.imshow(temperature_history[-1], cmap='rainbow', extent=[0, plate_size, 0, plate_size])
plt.colorbar(label='Temperature (°C)')
plt.title(f'Temperature Distribution at t={total_time} s')
plt.xlabel('X-axis (m)')
plt.ylabel('Y-axis (m)')

  
# Create a figure for 3D temperature distribution
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(121, projection='3d')

# Create grid points for the 3D plot
X, Y = np.meshgrid(np.linspace(0, plate_size, num_points), np.linspace(0, plate_size, num_points))

# Plot the 3D temperature distribution at a specific time
ax.plot_surface(X, Y, temperature_history[-1], cmap='rainbow')
ax.set_xlabel('X-axis (m)')
ax.set_ylabel('Y-axis (m)')
ax.set_zlabel('Temperature (°C)')
ax.set_title(f'3D Temperature Distribution at t={total_time} s')

plt.tight_layout()
plt.show()

critical_points_x = []
critical_points_y = []

# Identify points with critical heat concentration using the bisection method
def bisection_critical_points(temperature_history, threshold):
    for t, temp in enumerate(temperature_history):
        if t == 0:
            continue

        diff = temperature_history[t] - temperature_history[t - 1]
        hot_spots = np.argwhere(diff > threshold)

        for hot_spot in hot_spots:
            critical_points_x.append(hot_spot[0] * dx)
            critical_points_y.append(hot_spot[1] * dy)

threshold_temp_change = 0.1  # Adjust this threshold as needed
bisection_critical_points(temperature_history, threshold_temp_change)


# Create a figure for 3D temperature distribution
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(121, projection='3d')

# Create grid points for the 3D plot
X, Y = np.meshgrid(np.linspace(0, plate_size, num_points), np.linspace(0, plate_size, num_points))

# Plot the 3D temperature distribution at a specific time
ax.plot_surface(X, Y, temperature_history[-1], cmap='rainbow')
ax.set_xlabel('X-axis (m)')
ax.set_ylabel('Y-axis (m)')
ax.set_zlabel('Temperature')
ax.set_title(f'3D Temperature Distribution at t={total_time} s')

# Create a scatter plot for critical points
ax.scatter(critical_points_x, critical_points_y, np.full(len(critical_points_x), 100), c='red', marker='x', s=50, label='Critical Points')
ax.legend()

plt.tight_layout()
plt.show()


