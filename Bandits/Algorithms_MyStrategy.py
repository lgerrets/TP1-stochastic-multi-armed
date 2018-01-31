import numpy as np

class UCB_3_1:
    def __init__(self, nbArms, alpha=3, maxReward=1.):
        self.A = nbArms
        self.Alpha = alpha
        self.MaxReward = maxReward # les rewards seront normalisées sur [0,1]
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)
        self.Time = 1

    def chooseArmToPlay(self):
        if self.Time<=self.A: # Draw each arm once
            return self.Time-1
        else:
            return np.argmax(self.Means+np.sqrt(self.Alpha*np.log(self.Time)/(2*self.NbPulls)))

    def receiveReward(self, arm, reward):
        reward = reward/self.MaxReward
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1
        self.Time += 1

    def name(self):
        return "UCB-"+str(self.Alpha)

class UCB_3_2:
    def __init__(self, nbArms, alpha=3, maxReward=1.):
        self.A = nbArms
        self.Alpha = alpha
        self.MaxReward = maxReward # les rewards seront normalisées sur [0,1]
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)
        self.Time = 1

    def chooseArmToPlay(self):
        if self.Time<=self.A: # Draw each arm once
            return self.Time-1
        else:
            return np.argmax(self.Means+np.sqrt(self.Alpha*np.log(self.Time)/(2*self.NbPulls)))

    def receiveReward(self, arm, reward):
        reward = reward/self.MaxReward
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1
        self.Time += 1

    def name(self):
        return "UCB-"+str(self.Alpha)