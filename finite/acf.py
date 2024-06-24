import numpy as np
from numpy.random import choice
from numpy.random import randint
from scipy.special import softmax
from collections import deque
import os
from env import CustomEnv
from tqdm import tqdm

H=100
size=10
dims=2
nS=size**dims
nA=3**dims
alpha=25
N=1000000 #no of episodes

ff=10
visited=np.full((H+1,nS//ff),-1)
x=np.zeros(H+1).astype(int)

#state feature
def feat(h,state):
    res=np.zeros(x[h])
    res[visited[h,state//ff]]=1
    return res

#actor feature
def phi(h,state,action):
    res=np.zeros(nA*nS//ff)
    res[action*nS//ff:action*nS//ff+x[h]]=feat(h,state)
    return res

#algorithm
#### get number of log files in log directory
logrd="data/acf/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = 0
current_num_files = next(os.walk(logrd))[2]
run_num = len(current_num_files)
log_r = open(logrd+str(run_num)+".csv","w")
log_r.write('episode\treward\tconstraint\n')

K=0.01

Low=-10000000 #P in projection operator

theta = [np.zeros(nA*nS//ff) for h in range(H)]
value = [np.zeros(0) for h in range(H+1)]#parameter v
wvalue = [np.zeros(0) for h in range(H+1)]#parameter w
gamma=0

returns=deque(maxlen=10000)
violations=deque(maxlen=10000)
env=CustomEnv()
for n in tqdm(range(N)):
    #step-sizes
    a=0.1/(n//10000+1)**0.55
    b=0.01/(n//10000+1)**0.8
    c=0.0001/(n//10000+1)**1
    
    rewards = []
    actions = []
    states  = []
    constraints = []
    actual=[]
    state = env.reset() #start state
    for h in range(H):
        if visited[h,state//ff]==-1:
            value[h]=np.concatenate((value[h],[0]))
            wvalue[h]=np.concatenate((wvalue[h],[0]))
            visited[h,state//ff]=x[h]
            x[h]+=1
        
        probs=softmax([np.dot(phi(h,state,k),theta[h]) for k in range(nA)])
        action=choice(nA,p=probs/np.sum(probs))

        #transition
        new_state,reward,_,constraint=env.step(action)

        states.append(state)
        actions.append(action)
        constraints.append(constraint)
        rewards.append(reward+gamma*constraint)
        actual.append(reward)

        state = new_state

    #terminal
    if visited[H,state//ff]==-1:
        value[H]=np.concatenate((value[H],[0]))
        wvalue[H]=np.concatenate((wvalue[H],[0]))
        visited[H,state//ff]=x[H]
        x[H]+=1
    reward,constraint=0,0
    
    states.append(state)
    constraints.append(constraint)
    rewards.append(reward+gamma*(constraint-alpha))
    actual.append(reward)

    #learning
    for h in range(H):
        probs=softmax([np.dot(phi(h,states[h],k),theta[h]) for k in range(nA)])
        delta=rewards[h]+np.dot(feat(h+1,states[h+1]),value[h+1])-np.dot(feat(h,states[h]),value[h])
        wdelta=constraints[h]+np.dot(feat(h+1,states[h+1]),wvalue[h+1])-np.dot(feat(h,states[h]),wvalue[h])
        psi=phi(h,states[h],actions[h])-sum([phi(h,states[h],k)*probs[k] for k in range(nA)])
        value[h]+=a*delta*feat(h,states[h])
        wvalue[h]+=a*wdelta*feat(h,states[h])
        theta[h]+=b*psi*delta
        #terminal
    delta=rewards[H]-np.dot(feat(H,states[H]),value[H])
    value[H]+=a*delta*feat(H,states[H])
    wdelta=constraints[H]-alpha-np.dot(feat(H,states[H]),wvalue[H])
    wvalue[H]+=a*wdelta*feat(H,states[H])

    #lagrangian update
    gamma=min(0,gamma-c*(np.dot(feat(0,states[0]),wvalue[0])))

    returns.append(np.sum(actual))
    violations.append(np.sum(constraints))

    if n%10000==0:
        #print(probs)
        mean=np.mean(returns)
        cmean=np.mean(violations)
        print(n//10000,mean,cmean,gamma)
        log_r.write(str(n)+'\t'+str(mean)+'\t'+str(cmean)+'\n')
        log_r.flush()

log_r.close()

