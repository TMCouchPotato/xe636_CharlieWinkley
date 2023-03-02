import torch
import torch.nn as nn
import pygame
import numpy as np
import sys

pygame.init()
pygame.font.init()

DISPLAY_WIDTH = 500
DISPLAY_HEIGHT = 300

gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH,DISPLAY_HEIGHT))
pygame.display.set_caption('NN Graph')
clock = pygame.time.Clock()
#Setting sizes of layers within the structure
nIn, nH, nOut, batchSize = 10, 5, 1, 10
#Creating a randomised input
x = torch.randn(batchSize, nIn)
#creating a target output
y = torch.tensor([[1.0], [0.0], [0.0],
[1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
#Constructing the model (Linear Activation using nIn input features and nH output features, passed through a ReLU
#activation function. This is passed through another Linear activation layer with no. input features nH and no. output
#features nOut, passed through a sigmoid activation function for the output
model = nn.Sequential(nn.Linear(nIn, nH),
                     nn.ReLU(),
                     nn.Linear(nH, nOut),
                     nn.Sigmoid())
#Describes the loss function (Mean Squared Error in this case)
criterion = torch.nn.MSELoss()
#optimiser for updating model based on calculated loss.
optimiser = torch.optim.SGD(model.parameters(), lr = 0.01)
print(model._modules['0'].weight)
print(model._parameters.)
gameExit = False
def drawTree(model):
    longestLayer = 0
    layerCount = 0
    layerEst = len(model._modules)
    layerList = []
    for i in range(layerEst):
        if hasattr(model._modules[str(i)], "weight"):
            if len(model._modules[str(i)].weight)>longestLayer:
                longestLayer = len(model._modules[str(i)].weight)
                layerList.append(model._modules[str(i)].weight)
                layerCount=layerCount+1
    print(longestLayer, "  then ", layerCount)
    treeMatrix = np.ndarray((longestLayer, layerCount, 2))
    xPixChange = DISPLAY_WIDTH/(layerCount+1)
    yPixChange = DISPLAY_HEIGHT/(longestLayer+1)
    print(treeMatrix)
    for i in range(layerCount):
        if len(layerList[i])<longestLayer:
            for j in range(len(layerList[i])):
                treeMatrix[i][j] = (xPixChange*(i+1),(DISPLAY_HEIGHT/(len(layerList[i])+1))*(j+1))
        else:
            for j in range(len(layerList[i])):
                treeMatrix[i][j][0] = xPixChange*(i+1)
                treeMatrix[i][j][1] = (yPixChange * (j + 1))
    print("Tree Matrix ", treeMatrix)
while not gameExit:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Looks for quitting event in every iteration (Meaning closing the game window)
            gameExit = True
    x = torch.randn(batchSize, nIn)
    # model training loop
    for epoch in range(50):
        # apply the model to the given input
        yPred = model(x)
        # apply the loss function to the predicted values given the resulting values, returning in the loss values.
        loss = criterion(yPred, y)
        # print the iteration and average loss
        print('epoch: ', epoch, ' loss: ', loss.item())
        # reset the optimiser gradients
        optimiser.zero_grad()
        # find the gradients of the loss value to the predicted value.
        loss.backward()
        # make a step based on the calculated gradients
        optimiser.step()
    pygame.display.update()  # Pygame display update
    drawTree(model)
    clock.tick(5)
