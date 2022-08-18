import math
import numpy as np
import matplotlib.pyplot as plt
import torch

path = "path to directory with frames"

def draw(point, equation, figure, history):
    if not len(history):
        figure.plot_surface(x, y, equation, rstride=8, cstride=8, alpha=0.3, color="green")
        figure.contour(x, y, equation)
        figure.scatter(point[0], point[1], point[2], color="red")
        
    else:
        figure.plot_surface(x, y, equation,  rstride=8, cstride=8, alpha=0.3, color="green")
        figure.contour(x, y, equation)

        #this is heatmap, yo use it comment 2 previous lines
        #figure.contourf(x, y, equation, zdir='z', offset=-100, cmap=cm.coolwarm)
        
        counter = 0
        for i in history:
            figure.scatter(i[0], i[1], i[2], color="red")

            #if you want to show an image you might have performance issues because the number of points, you can reduce it here
            #counter +=1
            #if(counter % 1 == 0):
                #figure.scatter(i[0], i[1], i[2], color="red")

def calcEquation(x, y):
    return 3*(x**2) + 3 * (y**2)

def df_dx(point):
    return 6 * point[0]

def df_dy(point):
    return  6 * point[1]

def calcGradient(point):
    newx = df_dx(point=point)
    newy = df_dy(point=point)
    gradient = [newx, newy, 0]
    gradient = torch.tensor(gradient)
    return gradient

#this probably exists but idk
def namef(path, fiter):
    if fiter < 10:
        return path + "\\00" + str(fiter)
    elif fiter >= 10 and fiter < 100:
        return path + "\\0" + str(fiter)
    else:
        return path + "\\" + str(fiter)


x = np.linspace(-35, 35, 120)
y = np.linspace(-35, 35, 120)
x, y = np.meshgrid(x, y)

#####
EQUATION = 3*(x**2) + 3 * (y**2)
#####

point = [20, 30, 3900]
point = torch.tensor(point)
points = []

learning_rate0 = 1.0e-3
learning_rate = 1.0e-3
decay = 1.09

Amount_of_frames = 260

for i in range (1, Amount_of_frames):
    #grpahics idk
    graph = plt.figure().gca(projection='3d')
    draw(point, EQUATION, graph, points)

    #setting a name for the file
    name = namef(path, i)
    plt.imsave(name)

    if i % 10 == 0:
        print("Epoch {}".format(i))
    points.append(point)
    plt.close()

    #gradeint descent
    point = point - learning_rate * calcGradient(point=point)
    
    #new point
    newpoint = [point[0], point[1], calcEquation(point[0], point[1])]
    newpoint = torch.tensor(newpoint)
    point = newpoint

    #learning rate step decay
    learning_rate = learning_rate0 * (decay**math.floor((1 + i)/10))

    

