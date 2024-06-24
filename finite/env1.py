import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Discrete,MultiDiscrete

if __name__=="__main__":
    H=100
    size=10
    dims=2
    nS=size**dims
    nA=3**dims
    save1=np.random.randint(nS,size=(H,10))
    save2=np.random.randint(nS,size=(H,10))
    np.savetxt("mdp/S1",save1)
    np.savetxt("mdp/S2",save2)


class CustomEnv(gym.Env):
    def __init__(self):
        save1=np.loadtxt("mdp/S1").astype(int)
        save2=np.loadtxt("mdp/S2").astype(int)
        self.H=100
        self.size=10
        self.dims=2
        self.nS=self.size**self.dims
        self.nA=3**self.dims
        self.observation_space=MultiDiscrete([self.size]*self.dims)
        self.action_space=Discrete(self.nA)
        self.R=np.zeros((self.H,self.nS))
        self.G=np.zeros((self.H,self.nS))

        for ti in range(self.H):
            self.R[ti,save1[ti]]=10
            self.G[ti,save2[ti]]=10
        
        self.P=np.ones((self.nA,self.nA))
        for i in range(self.nA):
            self.P[i,i]=9*(self.nA-1)
            self.P[i]/=np.sum(self.P[i])
        

    def reset(self,seed=None):
        np.random.seed(seed)
        start_state=0
        self.t=0
        self.state=np.ones(self.dims)*start_state
        return self.state,None

    def step(self,action):
        direction=np.random.choice(self.nA,p=self.P[action])
        for i in reversed(range(self.dims)):
            self.state[i]=(self.state[i]+direction%3-1)%self.size
            direction=direction//3
        state_index=0
        for i in range(self.dims):
            state_index=state_index*self.size+self.state[i]
        reward=self.R[self.t,int(state_index)]
        constraint=self.G[self.t,int(state_index)]
        self.t+=1
        terminated=self.t==self.H
        info={'cost':constraint}
        return self.state,reward,terminated,False,info

