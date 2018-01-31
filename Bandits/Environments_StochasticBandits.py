import numpy as np
import scipy.stats as ss

class StochasticBandit:
    def __init__(self,nbArms):
        self.A = nbArms
        self.armDistributions = []
        self.armMeans = []
        self.bestarm = -1
        self.maxReward=0.


    def createBernoulliArmsFromMeans(self,means):
        self.bestarm = np.argmax(means)
        for a in range(0,self.A):
            self.armDistributions.append( ss.bernoulli(means[a]) )
            self.armMeans.append(means[a])

    def createBernoulliArms(self,minGap):
        self.maxReward=1.
        maxMean = minGap + np.random.rand()*(1.-minGap)
        secondmaxMean= maxMean-minGap
        bestarm = np.random.randint(0,self.A)
        secondbestarm=0
        if(bestarm>0):
         secondbestarm = np.random.randint(0,bestarm)
        else:
         secondbestarm = np.random.randint(bestarm+1,self.A)
        for a in range(0,self.A):
            m = np.random.rand()*secondmaxMean
            self.armDistributions.append( ss.bernoulli(m) )
            self.armMeans.append(m)
        self.armDistributions[bestarm] = ss.bernoulli(maxMean)
        self.armMeans[bestarm] = maxMean
        self.armDistributions[secondbestarm] = ss.bernoulli(secondmaxMean)
        self.armMeans[secondbestarm] = secondmaxMean
        self.bestarm = bestarm

    def createGaussianArms(self,minGap,variance):
        self.maxReward=np.infty
        maxMean = minGap + np.random.rand()*(1.-minGap)
        secondmaxMean= maxMean-minGap
        bestarm = np.random.randint(0,self.A)
        secondbestarm=0
        if(bestarm>0):
         secondbestarm = np.random.randint(0,bestarm)
        else:
         secondbestarm = np.random.randint(bestarm+1,self.A)
        for a in range(0,self.A):
            m = np.random.rand()*secondmaxMean
            self.armDistributions.append( ss.norm(loc=m,scale=variance) )
            self.armMeans.append(m)
        self.armDistributions[bestarm] = ss.norm(loc=maxMean,scale=variance)
        self.armMeans[bestarm] = maxMean
        self.armDistributions[secondbestarm] = ss.norm(loc=secondmaxMean,scale=variance)
        self.armMeans[secondbestarm] = secondmaxMean
        self.bestarm = bestarm

    def GenerateReward(self,arm):
        reward = self.armDistributions[arm].rvs()
        expectedInstantaneousRegret =  self.armMeans[self.bestarm]-self.armMeans[arm]
        return reward,expectedInstantaneousRegret

    def name(self):
        s = ""
        for m in self.armMeans:
            s+= "_{:.2}".format(str(m*100))
        return  "Means"+ s