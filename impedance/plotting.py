# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 23:10:13 2025

@author: 13391
"""

import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load('losses_archiveImp250219.npz', allow_pickle=True)

#np.savez('losses_archiveImp.npz', losses_total=loss_arr_total, losses_U = loss_arr_U, losses_F=loss_arr_F, A0=A0_arr, A1 = A1_arr, C0=C0_arr, C1=C1_arr, D=D_arr)
# List all the arrays stored in the .npz file
print("Arrays in the .npz file:", data.files)

# Access individual arrays using their keys
A0 = data['A0']  # Replace 'array1' with the actual key
A1 = data['A1']  # Replace 'array2' with the actual key
C0 = data['C0']  # Replace 'array1' with the actual key
C1 = data['C1']  # Replace 'array2' with the actual key
A0 = data['A0']  # Replace 'array1' with the actual key
D = data['D']  # Replace 'array2' with the actual key

# Print the arrays
# print("Array 1:", array1)
#%% print("Array 2:", array2)

# Close the .npz file (optional, but good practice)
#data.close()
plt.figure()
plt.plot(A0, label='A0')
plt.plot(A1, label='A1')
plt.plot(C0, label='C0')
plt.plot(C1, label='C1')
plt.plot(D, label='D')
# Adding title and labels
plt.title('Plot of 5 Arrays')
plt.xlabel('Index')
plt.ylabel('Value')
# Adding a legend
plt.legend()
plt.show()