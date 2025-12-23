import numpy as np
import matplotlib.pyplot as plt
 
# ==========================================
# 1. Define Physical Constants for 18650 Cell
# ==========================================
# These values are typical for a commercial cell like a Panasonic NCR18650B
m = 0.0465        # Mass of the cell (kg)
Cp = 950          # Specific heat capacity (J/kg.K)
R_int = 0.045     # Internal DC resistance (Ohms) - averaged
h = 15            # Convective heat transfer coefficient (W/m^2.K) 
                  # (~5-10 for still air, ~15-25 for forced air)
D = 0.018         # Diameter (m)
L = 0.065         # Length (m)
A = np.pi * D * (L + D/2)  # Total Surface Area (m^2) approx. 0.0042 m^2
T_amb = 25        # Ambient temperature (°C)
T_safe = 60       # Safety limit temperature (°C)

# Thermal Time Constant
tau = (m * Cp) / (h * A)
print(f"Thermal Time Constant (tau): {tau:.1f} seconds")

# ==========================================
# 2. Generate 2D Plot: Temperature vs. Time
# ==========================================
print("Generating 2D Transient Response Plot...")
fig_2d = plt.figure(figsize=(10, 6))
time_secs = np.linspace(0, 3600, 200) # 1 hour
time_mins = time_secs / 60

# Test currents to plot
currents_to_test = [2, 4, 6, 8, 10] # Amps

for I in currents_to_test:
    # Calculate Heat Generation
    Q_gen = (I**2) * R_int
    # Steady-state temperature rise
    T_rise_ss = Q_gen / (h * A)
    # Analytical Solution for Temp vs Time
    T_cell = T_amb + T_rise_ss * (1 - np.exp(-time_secs / tau))
    
    plt.plot(time_mins, T_cell, label=f'{I}A Discharge')

# Add Safety Limit Line
plt.axhline(y=T_safe, color='r', linestyle='--', linewidth=2, label=f'Safety Limit ({T_safe}°C)')

# Formatting
plt.title('18650 Cell Temperature Rise Over Time (Various Currents)', fontsize=14)
plt.xlabel('Time (Minutes)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.xlim(0, 60)
plt.ylim(T_amb, 90)

# Save Figure
plt.savefig('battery_thermal_2D_plot.png', dpi=300)
plt.close(fig_2d)


# ==========================================
# 3. Generate 3D Plot: Thermal Envelope
# ==========================================
print("Generating 3D Thermal Envelope Plot...")
fig_3d = plt.figure(figsize=(14, 10))
ax = fig_3d.add_subplot(111, projection='3d')

# Create grids for Current and Time
I_range = np.linspace(1, 12, 60)     # 1A to 12A
t_range_mins = np.linspace(0, 60, 60) # 0 to 60 minutes
I_grid, t_grid_mins = np.meshgrid(I_range, t_range_mins)
t_grid_secs = t_grid_mins * 60

# Calculate Surface Temperature
Q_gen_grid = (I_grid**2) * R_int
T_rise_ss_grid = Q_gen_grid / (h * A)
T_surface = T_amb + T_rise_ss_grid * (1 - np.exp(-t_grid_secs / tau))

# Plot the Temperature Surface
surf = ax.plot_surface(I_grid, t_grid_mins, T_surface, cmap='jet', 
                       edgecolor='none', alpha=0.85, rstride=2, cstride=2)

# Create and Plot the Safety Limit Plane (Red, Transparent)
I_plane, t_plane = np.meshgrid(np.array([1, 12]), np.array([0, 60]))
T_plane = np.full_like(I_plane, T_safe)
ax.plot_surface(I_plane, t_plane, T_plane, color='red', alpha=0.3)

# Formatting
ax.set_title('18650 Thermal Safety Envelope', fontsize=16, pad=20)
ax.set_xlabel('Discharge Current (A)', fontsize=12, labelpad=10)
ax.set_ylabel('Time (Minutes)', fontsize=12, labelpad=10)
ax.set_zlabel('Temperature (°C)', fontsize=12, labelpad=10)
ax.set_zlim(T_amb, 100)

# Add Colorbar
cbar = fig_3d.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
cbar.set_label('Cell Temperature (°C)', fontsize=12)

# Set view angle for better perspective
ax.view_init(elev=30, azim=-130)

# Save Figure
plt.savefig('battery_thermal_3D_envelope.png', dpi=300)
plt.close(fig_3d)

print("\nSuccess! The following images have been generated:")
print("- battery_thermal_2D_plot.png")
print("- battery_thermal_3D_envelope.png")