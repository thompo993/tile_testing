import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

voltage = [700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800]

lhs_peak_vals = [0.06832, 0.08107]
lhs_peak_vals_fit_error = [0.0008107, 0.00074]




plt.axhline(0.3, color='gray', linestyle='--', label='Target Peak Position (0.30V)')
plt.plot(voltage, lhs_peak_vals, 'o-', label='LHS PMT')
plt.plot(voltage, rhs_peak_vals, 's--', label='RHS PMT')
plt.xlabel('Voltage (V)')
plt.ylabel('Peak Position [A.U]')
plt.title('Peak Position vs Voltage for LHS and RHS PMTs')
plt.grid()
plt.legend()
plt.show()


plt.axhline(0.28, color='gray', linestyle='--', label='Target Peak Position (0.28V)')
plt.plot(voltage_fine, lhs_peak_vals_fine, 'o-', label='LHS PMT')
plt.plot(voltage_fine, rhs_peak_vals_fine, 's--', label='RHS PMT')
plt.axvline(789.8, color="orange", linestyle='--', label = "LHS PMT Selected Voltage (790V)")
plt.axvline(777,  linestyle='--', label = "RHS PMT Selected Voltage (777V)")

plt.xlabel('Voltage (V)')
plt.ylabel('Peak Position [A.U]')
plt.title('Peak Position vs Voltage for LHS and RHS PMTs')
plt.grid()
plt.legend()
plt.show()


plt.axhline(0.3, color='gray', linestyle='--', label = "Target Peak Postion selected in last calibration")
plt.axhline(0.3, color='gray', linestyle='--', label = "True Target Peak Postion selected in last calibration")
plt.axvline(768, color='blue', linestyle='--', label = "Actual voltage target selected with 768V (LHS PMT voltage)")
plt.axvline(771, color='orange', linestyle='--', label = "Actual voltage target selected with 771V (RHS PMT voltage)")
plt.plot(voltage_251113_lhs, lhs_peak_vals_251113, 'o-', label='LHS PMT')
plt.plot(voltage_251113_rhs, rhs_peak_vals_251113, 's--', label='RHS PMT')
plt.xlabel('Voltage (V)')
plt.ylabel('Peak Position [A.U]')
plt.title('Peak Position vs Voltage for LHS and PMT (13/01/26)')
plt.grid()
plt.legend()
plt.show()