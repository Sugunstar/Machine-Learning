import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
def compute_cost(x, y, m, w, b):
    cost = 0.0
    for i in range(m):
        fxi = w*x[i] + b
        cost = (y[i] - fxi)**2
    return (1/(2*m))*cost

def gradiant_decent(x,y,m,w,b):
    df_dm = 0.0
    df_db = 0.0
    for i in range(m):
        fxi = w*x[i] + b
        df_dm += (fxi - y[i])*x[i]
        df_db += (fxi - y[i])
    df_dm /= m
    df_db /= m
    return df_dm, df_db

my_need = ['LotArea', 'SalePrice']
df = pd.read_csv('train.csv', usecols=my_need)
#print(df)
x_train = np.array(df['LotArea'], dtype=float)
y_train = np.array(df['SalePrice'], dtype=float)
#print(x_train,'\n',y_train)
alpha = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

#feature skewing
x_train = np.log1p(df['LotArea'])
y_train = np.log1p(df['SalePrice'])

#plt.scatter(x, y)
#plt.title("log(area) vs log(price)")
#plt.grid(True)
#plt.show()

#feature scaleing
x_mean = x_train.mean()
y_mean = y_train.mean()
xmax = x_train.max()
ymax = y_train.max()
m=x_train.shape
m=m[0]
#print(m)
for i in range(m):
    x_train[i] = x_train[i]/xmax
    y_train[i] = y_train[i]/ymax


#print(x_train,'\n',y_train)
#plt.scatter(x_train, y_train)
#plt.title("area vs price")
#plt.grid(True)
#plt.show()


w = 0
b = 0
cost = []
itr = [i for i in range(1000)]
for i in range(1000):
    dw, db = gradiant_decent(x_train, y_train, m, w, b)
    w = w -  0.003*dw
    b = b - 0.003*db
    temp_cost = compute_cost(x_train, y_train, m, w, b)
    cost.append(temp_cost)

#plt.plot(itr, cost)
#plt.title("iterations vs cost_fn")
#plt.grid(True)
#plt.show()


