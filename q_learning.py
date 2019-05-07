from environment import MountainCar
import numpy as np
import random
import sys

def main(args):
    pass

def updateQ(Q ,learningRate, gamma, reward,   ):
    firstPart = (1-learningRate)*Q[s][a]
    secondPart = learningRate * (reward + gamma * maxQ(s,a) )
    result = firstPart + secondPart


    Q[s][a] = result
    return None

def qsaw(s,w):
    return np.dot(s,w)

def updataW(w, learningRate, state,b, gamma, reward, nextState):
    q = qsaw(state,w,b)
    q2 = qsaw(nextState,w,b)
    firstPart = learningRate*(q - (reward+ gamma*q2))
    secondPart = state
    result = w - firstPart*secondPart

    return result



    MountainCar.step()








def pickAction(epsilon):
    action = None
    r = random.random()
    if r <= epsilon:
        action = random.randint(0,2)

    elif r > epsilon:
        max = 0
        for i in [0,1,2]:
            guess = qsaw(state, w, )

    """
    np.random.seed(0)
    p = np.array([0.1, 0.0, 0.7, 0.2])
    index = np.random.choice([0, 1, 2, 3], p=p.ravel())
    if index == 3:
        action = 0
    elif index == 0 or index == 1 or index ==2:
        action = index
    """

    return action

if __name__ == "__main__":
    #main(sys.argv)
    print("code start")
    # command line
    # formatted input

    mode = sys.argv[1]  # #raw or tile
    #trainInput = readFile(trainInput)

    weightOut = sys.argv[2]
    #indexToWord = readFile(indexToWord)

    returnsOut = sys.argv[3]
    #indexToTag = readFile(indexToTag)


    episodes = sys.argv[4]
    #maximum of the length of an episode
    maxIterations = sys.argv[5]
    #value of epsilon-greedy
    epsilon = sys.argv[6]
    #discount factor
    gamma = sys.argv[7]
    #alpha
    learningRate = sys.argv[8]

    #######################################################################################
    #main
    MC = MountainCar(mode)

    w = np.zeros((MC.state_space + 1, 3))

    for i in range(int(episodes)):
        MC.reset()
        done = False
        while done==False:
            action = pickAction(float(epsilon))
            nextState, reward, done = MC.step(action)


            w = updataW(w, learningRate, MC.state, gamma, reward, nextState)

        print(w)








