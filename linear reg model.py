import numpy as np
from matplotlib import pyplot as plt

def params (x_train, y_train):
     sigma_x=np.sum(np.array(x_train))
     sigma_y=np.sum(np.array(y_train))
     x2=(list(map(lambda x:x**2, x_train)))
     sigma_x2=np.sum(np.array(x2))
     xy=np.array(list(map(lambda x,y:x*y, x_train, y_train)))
     sigma_xy=np.sum(xy)
     #variables needed sigma_x, sigma_y, sigma_x2, sigma_xy
     A=np.array([[len(x_train), sigma_x], [sigma_x, sigma_x2]])
     B=np.array([sigma_y, sigma_xy])
     solution = np.linalg.solve(A, B)
     print(solution)
