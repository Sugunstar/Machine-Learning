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
    return solution

if __name__ == "__main__":
     try:
          xt=list(map(int,input("enter space saperated features for the ml model\n").split()))
          yt=list(map(int,input("enter the space saperated targets for the ml model\n").split()))
          if(len(xt)!=len(yt)):
               sys.exit("the number of features is not equal to number of targets")
     except Exception as e:
          print("the following execption has occured:\ne")
     else:
          res=params(xt,yt)
          x_test=float(input("enter a value you want to test for....\nenter 0 to exit\n"))
          while(x_test!=0):
               print("expected result is: ", res[0]+(x_test*res[1]))
               x_test=float(input("enter a value you want to test for....\nenter 0 to exit\n"))

