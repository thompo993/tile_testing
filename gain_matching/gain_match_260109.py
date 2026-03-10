import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


voltage = [700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800]
lhs_peak_vals = [0.1328125,
                 0.14453125,
                 0.15234375,
                 0.1796875,
                 0.19921875,
                 0.21875,
                 0.2421875,
                 0.265625,
                 0.28125,
                 0.34375,
                 0.375
]

rhs_peak_vals = [0.12109375,
                 0.125,
                 0.152343750,
                 0.16015625,
                 0.1640625,
                 0.171875,
                 0.1953125,
                 0.2109375,
                 0.26171875,
                 0.265625,
                 0.3359375
]

print(len(voltage))
print(len(lhs_peak_vals))
print(len(rhs_peak_vals))
assert len(voltage) == len(lhs_peak_vals) == len(rhs_peak_vals)



voltage_fine = [765, 770, 775, 780, 785, 790]

lhs_peak_vals_fine = [0.251,
                      0.265,
                      0.273,
                      0.292,
                      0.308,
                      0.318     
]

rhs_peak_vals_fine = [0.2080,
                      0.2207,
                      0.241,
                      0.253,
                      0.2656,
                      0.280
]
                      
voltage_251113_lhs = [700, 720, 740, 760, 780]
voltage_251113_rhs = [700, 720, 740, 780]

lhs_peak_vals_251113 = [0.1329,
                       0.1797,
                       0.2305,
                       0.250,
                       0.329
]
rhs_peak_vals_251113 = [0.1282, 0.155, 0.1941, 0.302]




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