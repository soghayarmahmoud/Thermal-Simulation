# 🔋 18650 Lithium-Ion Battery Thermal Management System (BTMS)

A **numerical simulation framework** for analyzing the thermal behavior of a single 18650 lithium-ion battery cell under various discharge currents and cooling conditions. This project is designed for **engineering analysis, academic research, and portfolio demonstration**.

---

## 📌 Project Overview

As lithium-ion battery energy density continues to rise, **effective thermal management** becomes critical to:

* Prevent **thermal runaway**
* Reduce **capacity fade**
* Extend **battery lifetime**
* Ensure **operational safety**

This project implements a **Battery Thermal Management System (BTMS)** simulation based on the **Lumped Thermal Capacity Model**, enabling prediction of cell temperature over time by balancing:

* Internal heat generation (Joule heating)
* Convective heat dissipation to the environment

The tool allows engineers to evaluate thermal safety under different **discharge currents**, **ambient temperatures**, and **cooling strategies**.

---

## 🔬 Theoretical Framework

### 1️⃣ Energy Balance Equation

The thermal behavior of the battery is governed by the energy conservation principle:

$$
mc_p \frac{dT}{dt} = \dot{Q}*{gen} - \dot{Q}*{loss}
$$

Where:

* $m c_p \frac{dT}{dt}$ — Rate of internal energy change
* $\dot{Q}*{gen} = I^2 R*{int}$ — Heat generation due to internal resistance
* $\dot{Q}*{loss} = h A (T - T*{amb})$ — Convective heat loss to surroundings

> 🔎 **Assumption:** The battery temperature is spatially uniform (lumped-parameter model).

---

### 2️⃣ Fluid Dynamics & Convective Heat Transfer

Cooling effectiveness is evaluated using classical heat transfer correlations:

$$
Re = \frac{\rho v D}{\mu}
$$

$$
Nu = \frac{hD}{k}
$$

Where:

* $Re$ — Reynolds Number (flow regime indicator)
* $Nu$ — Nusselt Number (convective heat transfer efficiency)
* $h$ — Convective heat transfer coefficient

These relationships allow the model to represent:

* Natural air cooling
* Forced air cooling
* Liquid cooling scenarios

---

## 📊 Visual Analysis

### 🔹 Transient Thermal Response (2D)

This plot shows the **cell temperature evolution over a 60-minute discharge** at multiple current levels. The steady-state region highlights the balance between heat generation and heat dissipation.

📈 *Visual Insight:* Higher currents result in elevated steady-state temperatures and faster thermal rise.

![2D Thermal Response](assets/battery_thermal_2D_plot.png)

---

### 🔹 Thermal Safety Envelope (3D)

A **3D surface plot** mapping:

* Time
* Discharge Current
* Cell Temperature

The red reference plane represents the **60°C safety limit**. Any region above this plane indicates a **thermally unsafe operating condition**.

![3D Thermal Safety Envelope](assets/battery_thermal_3D_envelope.png)

---

## 🛠️ Implementation

### ✅ Prerequisites

* Python **3.8+**
* NumPy
* Matplotlib

---

### 📥 Installation

```bash
git clone https://github.com/yourusername/18650-thermal-sim.git
cd 18650-thermal-sim
pip install -r requirements.txt
```

---

### ▶️ Usage

Run the main simulation script to generate thermal analysis plots:

```bash
python generate_plots.py
```

All generated figures will be saved to the `assets/` directory.

---

## 📝 Technical Specifications (Reference Parameters)

| Parameter             | Symbol    | Value  | Unit     |
| --------------------- | --------- | ------ | -------- |
| Cell Mass             | $m$       | 0.0465 | kg       |
| Specific Heat         | $c_p$     | 950    | J/(kg·K) |
| Internal Resistance   | $R_{int}$ | 0.045  | Ω        |
| Surface Area          | $A$       | 0.0042 | m²       |
| Thermal Time Constant | $\tau$    | ~700   | s        |

---

## 🚀 Future Roadmap

* 🔗 **Multi-cell Modules**
  Simulate thermal coupling in a 10s3p battery pack

* 🧭 **Anisotropic Modeling**
  Different radial and axial thermal conductivities

* ❄️ **Cooling Strategy Comparison**
  Air vs Water-Glycol vs Phase Change Materials (PCM)

* ⚡ **Electro-Thermal Coupling**
  Integrating SOC-dependent heat generation

---

## 🤝 Contributing

Contributions are welcome and encouraged!

If you would like to:

* Improve Nusselt correlations
* Add electrochemical heat generation models
* Extend the simulation to battery packs

Please open an **issue** or submit a **pull request**.

---

## 📜 License

This project is released under the **MIT License** — free to use for academic, research, and commercial purposes.

---

### ⭐ If you find this project useful, consider giving it a star on GitHub!
