import numpy as np
from scipy.special import softmax
from env1 import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys
from tqdm import tqdm
import torch
from torch.distributions import Categorical
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import os

class discrete_policy(nn.Module):
    def __init__(self, nS, nH1, nH2, nA):
        super(discrete_policy, self).__init__()
        self.h1 = nn.Linear(nS, nH1)
        self.h2 = nn.Linear(nH1, nH2)
        self.out = nn.Linear(nH2, nA)

    def forward(self, x):
        x = torch.tanh(self.h2(torch.tanh(self.h1(x))))
        x = torch.softmax(self.out(x), dim=1)
        return x

class value_function(nn.Module):
    def __init__(self, nS, nH1, nH2, nA):
        super(value_function, self).__init__()
        self.h1 = nn.Linear(nS, nH1)
        self.h2 = nn.Linear(nH1, nH2)
        self.out = nn.Linear(nH2, nA)

    def forward(self, x):
        x = torch.tanh(self.h2(torch.tanh(self.h1(x))))
        x = self.out(x)
        return x

H=100
size=10
dims=2
nS=size**dims
nA=3**dims
alpha=25
gamma=0
K=10 #trajectories
hidden=10

policy = [discrete_policy(dims, hidden, hidden, nA) for h in range(H)]
value = [value_function(dims, hidden, hidden, 1) for h in range(H+1)]
wvalue = [value_function(dims, hidden, hidden, 1) for h in range(H+1)]
voptim = [torch.optim.Adam(value[h].parameters()) for h in range(H+1)]
woptim = [torch.optim.Adam(wvalue[h].parameters()) for h in range(H+1)]
poptim = [torch.optim.Adam(policy[h].parameters()) for h in range(H)]

logrd="data/acnn/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

N=1000000//K

env=[CustomEnv() for i in range(K)]
returns=deque(maxlen=10000//K)
violations=deque(maxlen=10000//K)
fr.write("timestep\treturn\tconstraint\n")
for n in tqdm(range(N)):
    c=0.01
    states=torch.zeros((H+1,K,dims))
    actions=torch.zeros((H,K))
    rewards=torch.zeros((H+1,K))
    constraints=torch.zeros((H+1,K))
    actual=torch.zeros((H+1,K))

    state=np.zeros((K,dims))
    for i in range(K): state[i],_=env[i].reset()
    reward=np.zeros(K)
    constraint=np.zeros(K)
    for h in range(H):
        states[h]=torch.FloatTensor(state)
        probs=policy[h](torch.FloatTensor(state))
        actions[h]=Categorical(probs).sample()
        for i in range(K): 
            state[i],reward[i],_,_,info=env[i].step(int(actions[h,i]))
            constraint[i]=info['cost']
        rewards[h]=torch.FloatTensor(reward+gamma*constraint)
        constraints[h]=torch.FloatTensor(constraint)
        actual[h]=torch.FloatTensor(reward)

    reward,constraint=np.zeros(K),np.zeros(K)
    states[H]=torch.FloatTensor(state)
    rewards[H]=torch.FloatTensor(reward+gamma*(constraint-alpha))
    constraints[H]=torch.FloatTensor(constraint)
    actual[H]=torch.FloatTensor(reward)

    for h in range(H):
        probs=policy[h](states[h])
        delta=rewards[h]+value[h+1](states[h+1]).detach()-value[h](states[h])
        wdelta=constraints[h]+wvalue[h+1](states[h+1]).detach()-wvalue[h](states[h])
        vloss=torch.mean(0.5*delta**2)
        wloss=torch.mean(0.5*wdelta**2)
        ploss=torch.mean(-Categorical(probs).log_prob(actions[h])*delta.detach())
        voptim[h].zero_grad()
        vloss.backward()
        voptim[h].step()
        woptim[h].zero_grad()
        wloss.backward()
        woptim[h].step()
    
        poptim[h].zero_grad()
        ploss.backward()
        poptim[h].step()
    
    delta=rewards[H]-value[H](states[H])
    wdelta=constraints[H]-alpha-wvalue[H](states[H])
    vloss=torch.mean(0.5*delta**2)
    wloss=torch.mean(0.5*wdelta**2)
    voptim[H].zero_grad()
    vloss.backward()
    voptim[H].step()
    woptim[H].zero_grad()
    wloss.backward()
    woptim[H].step()

    gamma=min(0,gamma-c*torch.mean(wvalue[0](states[0])).item())

    returns.append(torch.sum(torch.mean(actual,1)))
    violations.append(torch.sum(torch.mean(constraints,1)))
    if n%(10000//K)==0:
        mean=np.mean(returns)
        cmean=np.mean(violations)
        fr.write(str(n)+"\t"+str(mean)+"\t"+str(cmean)+"\n")
        fr.flush()
        print(n//(10000//K),":",mean,cmean,gamma)
fr.close()




