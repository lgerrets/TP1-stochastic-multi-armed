import pylab as pl
import numpy as np
from copy import deepcopy

def OneBanditOneLearnerOneRun(bandit, learner, timeHorizon):
    arms= []
    rewards = []
    regrets = []
    cumulativeregrets = []
    cumulativeregret =0
    for t in range(0,timeHorizon):
        arm = learner.chooseArmToPlay()
        reward,expectedInstantaneousRegret=bandit.GenerateReward(arm)
        learner.receiveReward(arm,reward)
        # Update statistics
        arms.append(arm)
        rewards.append(reward)
        regrets.append(expectedInstantaneousRegret)
        cumulativeregret = cumulativeregret+expectedInstantaneousRegret
        cumulativeregrets.append(cumulativeregret)
    return arms,rewards,regrets,cumulativeregrets

def plotOneBanditOneLearnerOneRun(name, arms, rewards, regrets, cumulativeregrets, show=True):
    pl.figure(1)
    pl.clf()
    pl.xlabel("Arms", fontsize=16)
    pl.ylabel("Arm histogram", fontsize=16)
    pl.hist(arms, max(arms) + 1)
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Arm histogram"+ '.pdf')

    pl.figure(2)
    pl.clf()
    pl.xlabel("Time steps", fontsize=16)
    pl.ylabel("Instantaenous rewards", fontsize=16)
    pl.plot(range(0, len(rewards)), rewards, 'black', linewidth=0, marker='.', markeredgewidth=1,
            markerfacecolor='none', markersize=1)
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Instantaenous rewards"+ '.pdf')

    pl.figure(3)
    pl.clf()
    pl.xlabel("Regret values", fontsize=16)
    pl.ylabel("Instantaenous Regret histogram", fontsize=16)
    pl.hist(regrets, 50)
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Instantaenous Regret histogram"+ '.pdf')

    pl.figure(4)
    pl.clf()
    pl.xlabel("Time steps", fontsize=16)
    pl.ylabel("Cumulative regret", fontsize=16)
    pl.plot(range(0, len(cumulativeregrets)), cumulativeregrets, 'black', linewidth=1, marker='.', markeredgewidth=1,
            markerfacecolor='none', markersize=4)
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Cumulative regret"+ '.pdf')

def OneBanditOneLearnerNRuns(bandit, learner, timeHorizon, n):
    all_cumulativeregrets = np.zeros((timeHorizon,n))
    for i in range(n):
        learner.clear()
        arms,rewards,regrets,cumulativeregrets = OneBanditOneLearnerOneRun(bandit, learner, timeHorizon)
        all_cumulativeregrets[:,i] = cumulativeregrets
       # learner.clear()
        #for t in range(0,timeHorizon):
         #   arm = learner.chooseArmToPlay()
          #  reward,expectedInstantaneousRegret=bandit.GenerateReward(arm)
           # learner.receiveReward(arm,reward)
            # Update statistics
            #cumulativeregret = cumulativeregret+expectedInstantaneousRegret
            #all_cumulativeregrets[t,i] = cumulativeregret
    return all_cumulativeregrets

def plotOneBanditOneLearnerNRuns(all_cumulativeregrets):
    all_cumulativeregrets = np.array(all_cumulativeregrets)
    timeHorizon,n = all_cumulativeregrets.shape
    
    pl.figure(1)
    pl.clf()
    pl.xlabel("Cumulative regrets", fontsize=16)
    pl.ylabel("Cumulative regrets histogram", fontsize=16)
    pl.hist(all_cumulativeregrets[timeHorizon-1,:], facecolor='green', alpha=0.75)
    pl.show()
    
    averageregret = np.mean(all_cumulativeregrets,axis=1)
    pl.figure(2)
    pl.clf()
    pl.xlabel("Time steps", fontsize=16)
    pl.ylabel("Mean of cumulative regrets", fontsize=16)
    pl.plot(np.arange(0,timeHorizon,1),averageregret)
    pl.show()

def OneBanditNLearnersNRuns(bandit, learners, timeHorizon, n):
    m = len(learners)
    all_cumulativeregrets = np.zeros((timeHorizon,n,m))
    for i in range(m):
        all_cumulativeregrets[:,:,i] = OneBanditOneLearnerNRuns(bandit, learners[i], timeHorizon, n)
    return all_cumulativeregrets

def plotOneBanditNLearnersNRuns(all_cumulativeregrets,learners):
    timeHorizon,n,m = all_cumulativeregrets.shape
    averageregret = np.mean(all_cumulativeregrets,axis=1)
    pl.figure(2)
    pl.clf()
    pl.xlabel("Time steps", fontsize=16)
    pl.ylabel("Mean of cumulative regrets", fontsize=16)
    x = np.arange(0,timeHorizon,1)
    for i in range(m):
        pl.plot(x,averageregret[:,i],label=learners[i].name())
    pl.legend()
    pl.show()
