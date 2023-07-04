
import sys
from tkinter.tix import Tree
import numpy as np
import matplotlib.pyplot as plt

def create_polynomial_data(x, poly_coeffs, add_noise):
    ynoise = np.random.normal(0, 2, len(x))
    #print(ynoise)
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

def solve_by_pseudo_inv(x,y, poly_order):
    A = np.zeros((len(x),poly_order+1))
    for i in range(0,len(x)):
        for j in range(0,poly_order+1):
            A[i,j] = x[i]**(j)
    print(A)
    # compute pseudo inverse of A
    u, s, vt = np.linalg.svd(A)
    # print different components
    print("U: ", u)
    print("Singular Values", s)
    print("V^{T}", vt)

    #inverse of diagonalmatrix is the reciprocal of each element
    s_inv = 1.0 / s
    sz = np.zeros(A.shape)
    sz[0:len(s),0:len(s)] = np.diag(s_inv) # pad rows of zeros
    sinv = sz
    print(sinv)
    # calculate pseudoinverse
    pseudo_inv = np.dot(np.dot(vt.T, sinv.T), u.T)
    print(pseudo_inv) # 9x4 in our example
    beta_coeffs = np.dot(pseudo_inv,y)
    return beta_coeffs

def main():
    x = [0,0.5,1,1.5,2,2.5,3,3.5,4]
    x = np.asarray(x, float)
    poly_coeffs = [1, 1, 1, 1] # 2x^3 - 9x^2 + 8x + 5
    y, noisy_y = create_polynomial_data(x,poly_coeffs, True) # True means add noise
    print(y)
    plot_data(x, y, noisy_y, y)
    beta_coeffs = solve_by_pseudo_inv(x,noisy_y, 3)
    print('---------beta_coeffs--------')
    print(beta_coeffs)
    # np.flip reverses the coefficient array
    yfitted, noisy_y2 = create_polynomial_data(x,np.flip(beta_coeffs), False) # false means do not add noise
    
    plot_data(x, yfitted, noisy_y, y) # noisy_y was the data we fitted
if __name__ == "__main__":
 sys.exit(int(main() or 0))