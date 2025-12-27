import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

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
T_runaway = 150   # Thermal runaway initiation temperature (°C)

# Electrical Properties
Q_nominal = 3.4   # Nominal capacity (Ah)
V_nominal = 3.6   # Nominal voltage (V)
V_max = 4.2       # Maximum voltage (V)
V_min = 2.5       # Minimum voltage (V)
V_cutoff = 3.0    # Cutoff voltage (V)

# Aging Parameters
cycle_life_nominal = 500  # Cycles to 80% capacity at 1C
fade_rate = 0.0004  # Capacity fade per cycle (0.04% per cycle)

# Thermal Time Constant
tau = (m * Cp) / (h * A)
print(f"Thermal Time Constant (tau): {tau:.1f} seconds")
print(f"Cell Nominal Energy: {Q_nominal * V_nominal:.2f} Wh")
print(f"Cell Nominal Power (1C): {Q_nominal * V_nominal:.2f} W\n")

# ==========================================
# 2. Temperature vs. Time (Original 2D Plot)
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
plt.title('18650 Cell Temperature Rise Over Time (Various Currents)', fontsize=14, fontweight='bold')
plt.xlabel('Time (Minutes)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.xlim(0, 60)
plt.ylim(T_amb, 90)

# Save Figure
plt.savefig('battery_thermal_2D_plot.png', dpi=300, bbox_inches='tight')
plt.close(fig_2d)


# ==========================================
# 3. 3D Thermal Envelope (Original)
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
ax.set_title('18650 Thermal Safety Envelope', fontsize=16, fontweight='bold', pad=20)
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
plt.savefig('battery_thermal_3D_envelope.png', dpi=300, bbox_inches='tight')
plt.close(fig_3d)


# ==========================================
# 4. NEW: State of Charge (SOC) Dynamics
# ==========================================
print("Generating SOC Dynamics Plot...")
fig_soc, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

discharge_currents = [1, 2, 3.4, 5, 7]  # Including 1C rate (3.4A)
colors = plt.cm.viridis(np.linspace(0, 1, len(discharge_currents)))

for idx, I in enumerate(discharge_currents):
    # Time to full discharge (hours)
    t_discharge_hrs = Q_nominal / I
    time_hrs = np.linspace(0, t_discharge_hrs, 200)
    
    # SOC calculation (Coulomb counting)
    SOC = 100 * (1 - (I * time_hrs) / Q_nominal)
    
    # Voltage calculation (simplified OCV-SOC relationship + IR drop)
    # OCV approximation: linear between V_max and V_min
    OCV = V_min + (V_max - V_min) * (SOC / 100)
    V_terminal = OCV - I * R_int
    
    # C-rate
    C_rate = I / Q_nominal
    
    ax1.plot(time_hrs, SOC, label=f'{I:.1f}A ({C_rate:.1f}C)', 
             color=colors[idx], linewidth=2)
    ax2.plot(time_hrs, V_terminal, color=colors[idx], linewidth=2)

# SOC Plot
ax1.set_title('State of Charge vs. Time (Various Discharge Rates)', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (Hours)', fontsize=12)
ax1.set_ylabel('State of Charge (%)', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='best')
ax1.set_xlim(0, Q_nominal)
ax1.set_ylim(0, 100)

# Voltage Plot
ax2.set_title('Terminal Voltage vs. Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (Hours)', fontsize=12)
ax2.set_ylabel('Voltage (V)', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.axhline(y=V_cutoff, color='r', linestyle='--', linewidth=2, label=f'Cutoff ({V_cutoff}V)')
ax2.legend(loc='best')
ax2.set_xlim(0, Q_nominal)
ax2.set_ylim(2.0, 4.5)

plt.tight_layout()
plt.savefig('battery_SOC_dynamics.png', dpi=300, bbox_inches='tight')
plt.close(fig_soc)


# ==========================================
# 5. NEW: Ragone Plot (Power vs Energy)
# ==========================================
print("Generating Ragone Plot...")
fig_ragone = plt.figure(figsize=(10, 7))

C_rates = np.logspace(-1, 1, 50)  # 0.1C to 10C
discharge_currents_ragone = C_rates * Q_nominal

specific_energy = []
specific_power = []

for I in discharge_currents_ragone:
    # Discharge time
    t_discharge = Q_nominal / I
    
    # Average voltage during discharge (accounting for IR drop)
    V_avg = (V_max + V_min) / 2 - I * R_int
    
    # Energy delivered (Wh)
    E = V_avg * Q_nominal
    
    # Power (W)
    P = V_avg * I
    
    # Specific values (per kg)
    specific_energy.append(E / m)  # Wh/kg
    specific_power.append(P / m)   # W/kg

plt.loglog(specific_energy, specific_power, 'b-', linewidth=3, label='18650 Cell')

# Add markers for specific C-rates
marker_C_rates = [0.2, 0.5, 1, 2, 5]
for C in marker_C_rates:
    idx = np.argmin(np.abs(C_rates - C))
    plt.plot(specific_energy[idx], specific_power[idx], 'ro', markersize=10)
    plt.annotate(f'{C}C', (specific_energy[idx], specific_power[idx]), 
                xytext=(10, 10), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.title('Ragone Plot: Specific Power vs. Specific Energy', 
          fontsize=14, fontweight='bold')
plt.xlabel('Specific Energy (Wh/kg)', fontsize=12)
plt.ylabel('Specific Power (W/kg)', fontsize=12)
plt.grid(True, which='both', linestyle=':', alpha=0.7)
plt.legend(loc='best', fontsize=12)

plt.savefig('battery_ragone_plot.png', dpi=300, bbox_inches='tight')
plt.close(fig_ragone)


# ==========================================
# 6. NEW: Thermal Runaway Analysis
# ==========================================
print("Generating Thermal Runaway Analysis...")
fig_runaway = plt.figure(figsize=(12, 7))

time_runaway = np.linspace(0, 120, 500)  # 2 hours in minutes
currents_runaway = [8, 10, 12, 15]

for I in currents_runaway:
    # Heat generation
    Q_gen = (I**2) * R_int
    
    # Temperature-dependent heat generation (exponential increase near runaway)
    # Arrhenius-like behavior
    T_cell_runaway = np.zeros_like(time_runaway)
    T_cell_runaway[0] = T_amb
    
    dt = time_runaway[1] - time_runaway[0]  # time step in minutes
    
    for i in range(1, len(time_runaway)):
        # Base heat generation
        Q_base = Q_gen
        
        # Self-heating acceleration factor (exponential above 80°C)
        if T_cell_runaway[i-1] > 80:
            acceleration = np.exp((T_cell_runaway[i-1] - 80) / 20)
        else:
            acceleration = 1.0
        
        Q_total = Q_base * acceleration
        
        # Heat dissipation
        Q_loss = h * A * (T_cell_runaway[i-1] - T_amb)
        
        # Net heat accumulation
        dT = (Q_total - Q_loss) * dt * 60 / (m * Cp)
        
        T_cell_runaway[i] = T_cell_runaway[i-1] + dT
    
    plt.plot(time_runaway, T_cell_runaway, label=f'{I}A Discharge', linewidth=2)

# Safety lines
plt.axhline(y=T_safe, color='orange', linestyle='--', linewidth=2, 
            label=f'Safety Limit ({T_safe}°C)')
plt.axhline(y=T_runaway, color='red', linestyle='--', linewidth=2, 
            label=f'Runaway Onset ({T_runaway}°C)')

plt.title('Thermal Runaway Risk Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Time (Minutes)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='best')
plt.xlim(0, 120)
plt.ylim(T_amb, 200)

plt.savefig('battery_thermal_runaway.png', dpi=300, bbox_inches='tight')
plt.close(fig_runaway)


# ==========================================
# 7. NEW: Capacity Fade Over Cycle Life
# ==========================================
print("Generating Capacity Fade Analysis...")
fig_aging = plt.figure(figsize=(12, 7))

cycles = np.arange(0, 1000, 1)
C_rates_aging = [0.5, 1.0, 2.0, 3.0]

for C in C_rates_aging:
    # Capacity fade model (exponential + linear)
    # Higher C-rates accelerate aging
    fade_acceleration = C ** 0.5
    
    # Capacity retention (%)
    capacity_retention = 100 * np.exp(-fade_rate * fade_acceleration * cycles)
    
    # Add linear component for long-term degradation
    linear_fade = 0.01 * cycles  # 1% per 100 cycles
    capacity_retention -= linear_fade
    
    # Ensure it doesn't go below 0
    capacity_retention = np.maximum(capacity_retention, 0)
    
    plt.plot(cycles, capacity_retention, label=f'{C}C Cycling', linewidth=2)

# 80% capacity line (End of Life)
plt.axhline(y=80, color='r', linestyle='--', linewidth=2, 
            label='End of Life (80% Capacity)')

plt.title('Battery Capacity Fade Over Cycle Life', fontsize=14, fontweight='bold')
plt.xlabel('Cycle Number', fontsize=12)
plt.ylabel('Capacity Retention (%)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='best')
plt.xlim(0, 1000)
plt.ylim(0, 105)

plt.savefig('battery_capacity_fade.png', dpi=300, bbox_inches='tight')
plt.close(fig_aging)


# ==========================================
# 8. NEW: Multi-Cell Pack Thermal Distribution
# ==========================================
print("Generating Multi-Cell Pack Thermal Distribution...")
fig_pack, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Simulate a 4x5 cell pack (20 cells)
pack_rows = 4
pack_cols = 5

# Discharge current per cell
I_pack = 5  # Amps per cell

# Simulate temperature distribution (center cells run hotter)
# due to reduced convection
temp_distribution = np.zeros((pack_rows, pack_cols))

for i in range(pack_rows):
    for j in range(pack_cols):
        # Distance from edge (normalized)
        dist_from_edge = min(i, pack_rows-1-i, j, pack_cols-1-j)
        
        # Base temperature rise
        Q_gen = (I_pack**2) * R_int
        T_rise_base = Q_gen / (h * A)
        
        # Reduced cooling for interior cells (convection penalty)
        cooling_factor = 1.0 - 0.15 * dist_from_edge
        
        # Final temperature
        temp_distribution[i, j] = T_amb + T_rise_base / cooling_factor

# Plot 1: Heatmap
sns.heatmap(temp_distribution, annot=True, fmt='.1f', cmap='hot', 
            cbar_kws={'label': 'Temperature (°C)'}, ax=ax1,
            vmin=T_amb, vmax=T_safe+10)
ax1.set_title('Battery Pack Thermal Distribution (4x5 Array)', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Column', fontsize=12)
ax1.set_ylabel('Row', fontsize=12)

# Plot 2: 3D Surface
X, Y = np.meshgrid(np.arange(pack_cols), np.arange(pack_rows))
ax2 = fig_pack.add_subplot(122, projection='3d')
surf_pack = ax2.plot_surface(X, Y, temp_distribution, cmap='hot', 
                              edgecolor='black', alpha=0.8)
ax2.set_title('3D Temperature Profile', fontsize=14, fontweight='bold')
ax2.set_xlabel('Column', fontsize=12)
ax2.set_ylabel('Row', fontsize=12)
ax2.set_zlabel('Temperature (°C)', fontsize=12)
ax2.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.savefig('battery_pack_thermal_distribution.png', dpi=300, bbox_inches='tight')
plt.close(fig_pack)


# ==========================================
# 9. NEW: C-Rate Performance Envelope
# ==========================================
print("Generating C-Rate Performance Envelope...")
fig_crate = plt.figure(figsize=(12, 7))

C_rates_perf = np.linspace(0.1, 5, 50)
discharge_currents_perf = C_rates_perf * Q_nominal

# Calculate deliverable capacity at each C-rate
# Higher rates reduce deliverable capacity due to voltage drop
deliverable_capacity = []
avg_voltage = []
energy_efficiency = []

for I in discharge_currents_perf:
    # Voltage drop increases with current
    V_drop = I * R_int
    
    # Average voltage during discharge
    V_avg = (V_max + V_min) / 2 - V_drop
    
    # Capacity available before hitting cutoff voltage
    # Simplified: capacity reduces as V_avg approaches V_cutoff
    if V_avg > V_cutoff:
        capacity_factor = (V_avg - V_cutoff) / (V_nominal - V_cutoff)
        capacity_factor = np.clip(capacity_factor, 0, 1)
    else:
        capacity_factor = 0
    
    deliverable_capacity.append(Q_nominal * capacity_factor)
    avg_voltage.append(V_avg)
    
    # Energy efficiency (vs. nominal)
    E_delivered = V_avg * Q_nominal * capacity_factor
    E_nominal = V_nominal * Q_nominal
    energy_efficiency.append(100 * E_delivered / E_nominal)

# Create subplot
fig_crate, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Deliverable Capacity vs C-Rate
ax1.plot(C_rates_perf, deliverable_capacity, 'b-', linewidth=3)
ax1.fill_between(C_rates_perf, 0, deliverable_capacity, alpha=0.3)
ax1.set_title('Deliverable Capacity vs. C-Rate', fontsize=14, fontweight='bold')
ax1.set_xlabel('C-Rate', fontsize=12)
ax1.set_ylabel('Capacity (Ah)', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.set_xlim(0, 5)

# Plot 2: Average Voltage vs C-Rate
ax2.plot(C_rates_perf, avg_voltage, 'g-', linewidth=3)
ax2.axhline(y=V_cutoff, color='r', linestyle='--', linewidth=2, label='Cutoff Voltage')
ax2.set_title('Average Discharge Voltage vs. C-Rate', fontsize=14, fontweight='bold')
ax2.set_xlabel('C-Rate', fontsize=12)
ax2.set_ylabel('Voltage (V)', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()
ax2.set_xlim(0, 5)

# Plot 3: Energy Efficiency vs C-Rate
ax3.plot(C_rates_perf, energy_efficiency, 'r-', linewidth=3)
ax3.fill_between(C_rates_perf, 0, energy_efficiency, alpha=0.3, color='red')
ax3.set_title('Energy Efficiency vs. C-Rate', fontsize=14, fontweight='bold')
ax3.set_xlabel('C-Rate', fontsize=12)
ax3.set_ylabel('Efficiency (%)', fontsize=12)
ax3.grid(True, linestyle=':', alpha=0.7)
ax3.set_xlim(0, 5)
ax3.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('battery_crate_performance.png', dpi=300, bbox_inches='tight')
plt.close(fig_crate)


# ==========================================
# 10. NEW: Internal Resistance vs Temperature
# ==========================================
print("Generating Internal Resistance vs Temperature...")
fig_resistance = plt.figure(figsize=(10, 6))

# Temperature range
T_range = np.linspace(-20, 60, 100)

# Internal resistance temperature dependence (Arrhenius-like)
# R increases at low temperatures, decreases slightly at high temperatures
R_ref = R_int  # Reference at 25°C
T_ref = 25

# Simplified model: exponential increase below reference, slight decrease above
R_temp = np.zeros_like(T_range)
for i, T in enumerate(T_range):
    if T < T_ref:
        # Exponential increase at low temp
        R_temp[i] = R_ref * np.exp(0.02 * (T_ref - T))
    else:
        # Slight decrease at high temp
        R_temp[i] = R_ref * (1 - 0.003 * (T - T_ref))

plt.plot(T_range, R_temp * 1000, 'b-', linewidth=3)  # Convert to mOhms
plt.axvline(x=T_ref, color='g', linestyle='--', linewidth=2, 
            label=f'Reference Temp ({T_ref}°C)')
plt.axhline(y=R_ref * 1000, color='g', linestyle='--', linewidth=2, alpha=0.5)

plt.title('Internal Resistance vs. Temperature', fontsize=14, fontweight='bold')
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Internal Resistance (mΩ)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

plt.savefig('battery_resistance_temperature.png', dpi=300, bbox_inches='tight')
plt.close(fig_resistance)


# ==========================================
# 11. NEW: Power Loss and Efficiency Map
# ==========================================
print("Generating Power Loss and Efficiency Map...")
fig_efficiency = plt.figure(figsize=(14, 10))
ax_eff = fig_efficiency.add_subplot(111, projection='3d')

# Create grids for Current and SOC
I_eff_range = np.linspace(1, 10, 40)
SOC_range = np.linspace(10, 100, 40)
I_eff_grid, SOC_grid = np.meshgrid(I_eff_range, SOC_range)

# Calculate efficiency
# OCV varies with SOC
OCV_grid = V_min + (V_max - V_min) * (SOC_grid / 100)
V_terminal_grid = OCV_grid - I_eff_grid * R_int

# Power loss
P_loss_grid = (I_eff_grid ** 2) * R_int

# Efficiency (%)
P_output_grid = V_terminal_grid * I_eff_grid
P_input_grid = OCV_grid * I_eff_grid
efficiency_grid = 100 * (P_output_grid / P_input_grid)

# Plot
surf_eff = ax_eff.plot_surface(I_eff_grid, SOC_grid, efficiency_grid, 
                                cmap='RdYlGn', edgecolor='none', alpha=0.9)

ax_eff.set_title('Battery Efficiency Map (Current vs. SOC)', 
                 fontsize=16, fontweight='bold', pad=20)
ax_eff.set_xlabel('Discharge Current (A)', fontsize=12, labelpad=10)
ax_eff.set_ylabel('State of Charge (%)', fontsize=12, labelpad=10)
ax_eff.set_zlabel('Efficiency (%)', fontsize=12, labelpad=10)
ax_eff.set_zlim(80, 100)

# Colorbar
cbar_eff = fig_efficiency.colorbar(surf_eff, ax=ax_eff, shrink=0.6, aspect=15, pad=0.1)
cbar_eff.set_label('Efficiency (%)', fontsize=12)

ax_eff.view_init(elev=25, azim=-120)

plt.savefig('battery_efficiency_map.png', dpi=300, bbox_inches='tight')
plt.close(fig_efficiency)


# ==========================================
# Summary Report
# ==========================================
print("\n" + "="*60)
print("SUCCESS! Enhanced Battery Simulation Complete")
print("="*60)
print("\nGenerated Images:")
print("  1. battery_thermal_2D_plot.png - Temperature vs Time")
print("  2. battery_thermal_3D_envelope.png - 3D Thermal Safety Envelope")
print("  3. battery_SOC_dynamics.png - State of Charge & Voltage Dynamics")
print("  4. battery_ragone_plot.png - Power vs Energy (Ragone Plot)")
print("  5. battery_thermal_runaway.png - Thermal Runaway Analysis")
print("  6. battery_capacity_fade.png - Cycle Life & Aging")
print("  7. battery_pack_thermal_distribution.png - Multi-Cell Pack Thermal Map")
print("  8. battery_crate_performance.png - C-Rate Performance Envelope")
print("  9. battery_resistance_temperature.png - Internal Resistance vs Temp")
print(" 10. battery_efficiency_map.png - 3D Efficiency Map")
print("\n" + "="*60)
print("Key Equations Implemented:")
print("="*60)
print("• Thermal: Q = I²R, T(t) = T_amb + ΔT_ss(1-e^(-t/τ))")
print("• SOC: SOC(t) = SOC₀ - (I·t)/Q_nominal × 100%")
print("• Voltage: V_terminal = OCV(SOC) - I·R_internal")
print("• Power: P = V·I, Energy: E = V_avg·Q")
print("• Aging: C_retention = 100·e^(-k·cycles)")
print("• Efficiency: η = (V_terminal·I)/(OCV·I) × 100%")
print("="*60 + "\n")
