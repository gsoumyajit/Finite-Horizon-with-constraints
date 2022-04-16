import numpy as np
from numpy.random import choice
from numpy.random import randint
from scipy.special import softmax
from collections import deque
from random import sample
import os
from multiprocessing import Process

#generate MDP

T=100
size=10
nS=size*size
nA=4
alpha=20
N=50000 #no of episodes

P=np.zeros((T,nS,nA,nS))
R=np.zeros((T,nS,nA,nS))
G=np.zeros((T,nS,nA,nS))

save1=np.loadtxt("mdp/S1").astype(int)
save2=np.loadtxt("mdp/S2").astype(int)

def get(a,b):
    return (a%size)*size+b%size

def put(state):
    return (state//size,state%size)

#Transition matrix
for j in range(nS):
    a,b=put(j)
    indices=[get(a,b+1),get(a,b-1),get(a+1,b),get(a-1,b)]
    P[0,j,:,indices]=3
    for k in range(nA):
        P[0,j,k,indices[k]]=81
        P[:,j,k]=P[0,j,k]/np.sum(P[0,j,k])

#Reward matrix
for ti in range(20):
    for i in range(size):
        b=save1[ti,i]
        index=get(i,b)
        R[ti*5:(ti+1)*5,:,:,index]=10

#Constraint cost matrix
for ti in range(20):
    for i in range(size):
        b=save2[ti,i]
        index=[get(i,b)]
        G[ti*5:(ti+1)*5,:,:,index]=10

RT=np.zeros(nS)
GT=np.zeros(nS)

ff=5

#state feature
def feat(state):
    res=np.zeros(nS//ff)
    res[state//ff]=(state%ff+1)/ff
    return res

#actor feature
def phi(state,action):
    res=np.zeros(nA*nS//ff)
    res[action*nS//ff:(action+1)*nS//ff]=feat(state)
    return res

#algorithm
def algo2(constrained,seed):
    np.random.seed(seed)
    #### get number of log files in log directory
    logrd="fr"
    logcd="fc"
    run_num = 0
    current_num_files = next(os.walk(logrd))[2]
    run_num = len(current_num_files)
    log_r = open(logrd+"/PPO_fr_log_"+str(run_num)+".csv","w+")
    log_r.write('episode,reward\n')
    run_num = 0
    current_num_files = next(os.walk(logcd))[2]
    run_num = len(current_num_files)
    log_c = open(logcd+"/PPO_fc_log_"+str(run_num)+".csv","w+")
    log_c.write('episode,reward\n')
    K=5
    epsilon=0.01

    Low=-10000000 #P in projection operator

    theta = np.zeros((T,nA*nS//ff))
    value = np.zeros((T+1,nS//ff)) #parameter v
    Y=0
    gamma=0

    n=1
    J=np.zeros(N)
    S=np.zeros(N)

    returns=deque(maxlen=100)
    violations=deque(maxlen=100)

    beta=np.full(nS,1)
    beta[0]=(nS-1)*9
    beta=beta/np.sum(beta)
    while n<=N:
        #step-sizes
        a=K/(n**0.55)
        b=K/(n**0.8)
        c=K/(n**1)

        rewards = []
        actions = []
        states  = []
        constraints = []
        actual=[]

        state = choice(nS,p=beta) #start state

        for i in range(T):

            probs=softmax([np.dot(phi(state,k),theta[i]) for k in range(nA)])
            action=choice(nA,p=probs/np.sum(probs))

            #transition
            new_state=choice(nS,p=P[i,state,action])
            reward,constraint=R[i,state,action,new_state],G[i,state,action,new_state]

            states.append(state)
            actions.append(action)
            constraints.append(constraint)
            rewards.append(reward+gamma*constraint)
            actual.append(reward)

            state = new_state

        #terminal
        reward,constraint=RT[state],GT[state]

        states.append(state)
        constraints.append(constraint)
        rewards.append(reward+gamma*(constraint-alpha))
        actual.append(reward)

        #learning
        for i in range(T):
            probs=softmax([np.dot(phi(states[i],k),theta[i]) for k in range(nA)])
            delta=rewards[i]+np.dot(feat(states[i+1]),value[i+1])-np.dot(feat(states[i]),value[i])
            psi=phi(states[i],actions[i])-np.sum([phi(states[i],k)*probs[k] for k in range(nA)])
            value[i]+=a*delta*feat(states[i])
            theta[i]+=b*((psi*delta)-(epsilon*theta[i]))

        #terminal
        delta=rewards[T]-np.dot(feat(states[T]),value[T])
        value[T]+=a*delta*feat(states[T])

        #lagrangian update
        if(constrained):
            gamma=max(Low,min(0,gamma-c*(Y-alpha)))

        #recursion 25
        Y=(1-a)*Y + a*np.sum(constraints)

        returns.append(np.sum(actual))
        violations.append(np.sum(constraints))

        J[n-1]=np.mean(returns)
        S[n-1]=np.mean(violations)
        
        log_r.write('{},{}\n'.format(n,J[n-1]))
        log_r.flush()
        log_c.write('{},{}\n'.format(n,S[n-1]))
        log_c.flush()

        print("algo2:",n,":",J[n-1],S[n-1],gamma,np.sum(actual))
        n+=1

    log_r.close()
    log_c.close()
    print("Done")

if __name__ == '__main__':
    seed=randint(10000)
    algo2(False,seed)

