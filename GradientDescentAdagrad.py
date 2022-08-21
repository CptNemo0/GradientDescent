import numpy as np
import matplotlib.pyplot as plt
import torch

def draw(x, y, equation, figure, history):
    figure.plot_surface(x, y, equation,  rstride=8, cstride=8, alpha=0.3, color="green")
    figure.contour(x, y, equation)
        
    counter = 0
    for i in history:
        counter +=1
        if(counter % 5 == 0):
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
    retValue = [newx, newy]
    retValue = torch.tensor(retValue)
    retValue = retValue.to(torch.float32)

    return retValue

def main():
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    x, y = np.meshgrid(x, y)

    #####
    EQUATION = x**2 + y**2
    #####

    X = -15
    Y = 17

    learning_rate = 0.3

    point = [X, Y, X**2 + Y**2]
    point = torch.tensor(point)
    point = point.to(torch.float32)

    delta = [0.0, 0.0]
    delta = torch.tensor(delta)
    delta = delta.to(torch.float32)
    
    actuall_delta = torch.zeros(3)
    actuall_delta = actuall_delta.to(torch.float32)

    previous_squared_grad_sum = [0.0, 0.0]
    previous_squared_grad_sum = torch.tensor(previous_squared_grad_sum)
    previous_squared_grad_sum = previous_squared_grad_sum.to(torch.float32)

    points = []

    Amount_of_frames = 501

    for i in range (1, Amount_of_frames):

        points.append(point)
        gradient = calcGradient(point=point)

        if previous_squared_grad_sum.mean():
            delta = (-learning_rate * gradient) / (torch.sqrt(previous_squared_grad_sum))
        else:
            delta = -learning_rate * gradient

        actuall_delta = [delta[0], delta[1], 0]
        actuall_delta = torch.tensor(actuall_delta)
        actuall_delta = actuall_delta.to(torch.float32)

        previous_squared_grad_sum += gradient**2
        point += actuall_delta
        
        newpoint = [point[0], point[1], calcEquation(point[0], point[1])]
        newpoint = torch.tensor(newpoint)
        point = newpoint
        
    graph = plt.figure().gca(projection='3d')
    draw(x, y, EQUATION, graph, points)
    plt.show()

if __name__ == '__main__':
    main()

#Generaly speaking, Adagrad works best when used in a large scale operations. 
#That is the reason why it's performance is unimpressive, to say the least, in this example.