import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# second manual gain matching for the LHS and RHS PMTs
voltage = [700, 710, 720, 730, 740, 745, 750, 755, 760, 765, 770, 775, 780, 790, 800]
lhs_peak_vals = [0.068320437,
0.072117205,
0.095641624,
0.10579828260178893,
0.1167309591431769,
0.11984986373952768,
0.12534661892587687,
0.13017978569161812,
0.13747461618538942,
0.14391428896548014,
0.152697278,
0.15841353920324894,
0.16698826406684963,
0.18837941733280858,
0.19722504546384517,
]

lhs_peak_vals_err = [0.00080734,
0.004464046,
0.020242181,
0.000589741,
0.000876394,
0.000425797,
0.000959347,
0.002109632,
0.001819999,
0.001063123,
0.002272539,
0.003066349,
0.002103219,
0.005464679,
0.023109965
]

rhs_peak_vals = [0.064221837,
0.073553854,
0.077503828,
0.09213,
0.108641056,
0.11970525,
0.123630141,
0.131567518,
0.137584092,
0.145808775,
0.148689457,
0.155377391,
0.162387872,
0.176590055,
0.191983642
]

rhs_peak_vals_err = [0.00092431,
0.00151496,
0.00096197,
0.004841567,
0.00230258,
0.001429936,
0.001459678,
0.002191306,
0.002243751,
0.004196571,
0.004202584,
0.004395281,
0.002715975,
0.002597163,
0.006583222
]

lhs_peak_vals = np.array(lhs_peak_vals, dtype=np.float64)
rhs_peak_vals = np.array(rhs_peak_vals, dtype=np.float64)
lhs_peak_vals_err = np.array(lhs_peak_vals_err, dtype=np.float64)
rhs_peak_vals_err = np.array(rhs_peak_vals_err, dtype=np.float64)


expermimental_error_lhs = (6/100)*lhs_peak_vals
experimental_error_rhs = (6/100)*rhs_peak_vals

tot_lhs_error = (rhs_peak_vals_err**2 + expermimental_error_lhs**2)**0.5
tot_rhs_error = (rhs_peak_vals_err**2 + experimental_error_rhs**2)**0.5


# plt.axhline(0.3, color='gray', linestyle='--', label='Target Peak Position (0.30V)')
plt.errorbar(voltage, lhs_peak_vals, yerr=tot_lhs_error, fmt='o-', label='LHS PMT', capsize=5)
plt.errorbar(voltage, rhs_peak_vals, yerr=tot_rhs_error, fmt='o-', label='RHS PMT', capsize=5)
plt.xlabel('Voltage (V)')
plt.ylabel('Peak Position [A.U]')
plt.title('Peak Position vs Voltage for LHS and RHS PMTs')
plt.grid()
plt.legend(loc="upper left")
plt.show()


# plt.axhline(0.28, color='gray', linestyle='--', label='Target Peak Position (0.28V)')
# plt.plot(voltage_fine, lhs_peak_vals_fine, 'o-', label='LHS PMT')
# plt.plot(voltage_fine, rhs_peak_vals_fine, 's--', label='RHS PMT')
# plt.axvline(789.8, color="orange", linestyle='--', label = "LHS PMT Selected Voltage (790V)")
# plt.axvline(777,  linestyle='--', label = "RHS PMT Selected Voltage (777V)")

# plt.xlabel('Voltage (V)')
# plt.ylabel('Peak Position [A.U]')
# plt.title('Peak Position vs Voltage for LHS and RHS PMTs')
# plt.grid()
# plt.legend()
# plt.show()


# plt.axhline(0.3, color='gray', linestyle='--', label = "Target Peak Postion selected in last calibration")
# plt.axhline(0.3, color='gray', linestyle='--', label = "True Target Peak Postion selected in last calibration")
# plt.axvline(768, color='blue', linestyle='--', label = "Actual voltage target selected with 768V (LHS PMT voltage)")
# plt.axvline(771, color='orange', linestyle='--', label = "Actual voltage target selected with 771V (RHS PMT voltage)")
# plt.plot(voltage_251113_lhs, lhs_peak_vals_251113, 'o-', label='LHS PMT')
# plt.plot(voltage_251113_rhs, rhs_peak_vals_251113, 's--', label='RHS PMT')
# plt.xlabel('Voltage (V)')
# plt.ylabel('Peak Position [A.U]')
# plt.title('Peak Position vs Voltage for LHS and PMT (13/01/26)')
# plt.grid()
# plt.legend()
# plt.show()
