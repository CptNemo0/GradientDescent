import numpy as np
import matplotlib.pyplot as plt
import torch

def draw(x, y, point, equation, figure, history):
    if not len(history):
        figure.plot_surface(x, y, equation, rstride=8, cstride=8, alpha=0.3, color="green")
        figure.contour(x, y, equation)
        figure.scatter(point[0], point[1], point[2], color="red")
        
    else:
        figure.plot_surface(x, y, equation,  rstride=8, cstride=8, alpha=0.3, color="green")
        figure.contour(x, y, equation)
        
        counter = 0
        for i in history:
            counter +=1
            if(counter % 2 == 0):
                figure.scatter(i[0], i[1], i[2], color="red")

def calcEquation(x, y):
    return x**2 + y**2

def df_dx(point):
    return 2 * point[0]

def df_dy(point):
    return 2 * point[1]

def calcGradient(point):
    newx = df_dx(point=point)
    newy = df_dy(point=point)

    #retValue is gradient
    retValue = [newx, newy, 0]
    retValue = torch.tensor(retValue)
    retValue = retValue.to(float)

    return retValue

def main():
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    x, y = np.meshgrid(x, y)

    #####
    EQUATION = x**2 + y**2
    #####

    X = -10
    Y = 10

    learning_rate = 1.0e-2
    decay = 1.0e-3

    point = [X, Y, X**2 + Y**2]
    point = torch.tensor(point)
    point = point.to(torch.float32)

    delta = [0.0, 0.0, 0.0]
    delta = torch.tensor(delta)
    delta = delta.to(torch.float32)

    previous_delta = [0.0, 0.0, 0.0]
    previous_delta = torch.tensor(previous_delta)
    previous_delta = previous_delta.to(torch.float32)

    points = []

    Amount_of_frames = 200

    for i in range (1, Amount_of_frames):

        points.append(point)

        delta = decay * previous_delta - learning_rate * calcGradient(point=point)

        point += delta
        previous_delta = delta
        
        newpoint = [point[0], point[1], calcEquation(point[0], point[1])]
        newpoint = torch.tensor(newpoint)
        point = newpoint
        
    graph = plt.figure().gca(projection='3d')
    draw(x, y, point, EQUATION, graph, points)
    plt.show()

if __name__ == '__main__':
    main()