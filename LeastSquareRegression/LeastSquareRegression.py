

import sys
from tkinter.tix import Tree
import numpy as np
import matplotlib.pyplot as plt

def create_polynomial_data(x, poly_coeffs, add_noise):
    ynoise = np.random.normal(0, 2, len(x))
    #print(ynoise)
    #y = poly_coeffs[0]*x**4 + poly_coeffs[1]*x**3 + poly_coeffs[2]*x**2 + poly_coeffs[3]*x + poly_coeffs[4]
    y = poly_coeffs[0]*x**3 + poly_coeffs[1]*x**2 + poly_coeffs[2]*x + poly_coeffs[3]
    if add_noise == True:
        noisy_y = y + ynoise
    else:
        noisy_y = y
    return y, noisy_y

def plot_data(x, y, y2, yorig):
    area = 10
    colors =['black']
    plt.scatter(x, y, s=area, c=colors, alpha=0.5, linewidths=8)
    plt.title('Pseudo Inverse - Linear Least Squares Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    #plot the fitted line
    line,=plt.plot(x, y, '-', linewidth=2, label="y-fitted")
    line.set_color('red')
    line,=plt.plot(x, y2, '-', linewidth=2, label="y-data")
    line.set_color('blue')
    line,=plt.plot(x, yorig, '--', linewidth=2, label="orig")
    line.set_color('green')
    plt.legend(loc="upper left")
    plt.show()

def solve_by_inv(x,y, poly_order):
    A = np.zeros((poly_order+1,poly_order+1))
    Y = np.zeros((poly_order+1))
    for i in range(0,poly_order+1):
        for j in range(0,poly_order+1):
            sumA = 0
            for k in range(0,len(x)): # sum the data
                sumA = sumA + (x[k]**j)*(x[k]**i)
            A[i,j] = sumA
    print(A)
    for i in range(0,poly_order+1):
        sumy = 0
        for k in range(0,len(x)): # sum the data
            sumy = sumy + y[k] * (x[k]**i)
        Y[i] = sumy
    print(Y)
   
    # calculate inverse
    A_inv = np.linalg.inv(A)
    print(A_inv) # 9x4 in our example
    beta_coeffs = np.dot(A_inv,Y)
    return beta_coeffs

def main():
    x = [0,0.5,1,1.5,2,2.5,3,3.5,4]
    x = np.asarray(x, float)
    poly_coeffs = [2, -9, 8, 5] # 2x^3 - 9x^2 + 8x + 5
    y, noisy_y = create_polynomial_data(x,poly_coeffs, True) # True means add noise
    print(y)
    plot_data(x, y, noisy_y, y)
    beta_coeffs = solve_by_inv(x,noisy_y, 3)
    print('---------beta_coeffs--------')
    print(beta_coeffs)
    # np.flip reverses the coefficient array
    yfitted, noisy_y2 = create_polynomial_data(x,np.flip(beta_coeffs), False) # false means do not add noise
    
    plot_data(x, yfitted, noisy_y, y) # noisy_y was the data we fitted
if __name__ == "__main__":
 sys.exit(int(main() or 0))