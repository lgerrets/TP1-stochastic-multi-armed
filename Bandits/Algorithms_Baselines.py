import numpy as np
import Bandits.Algorithms_kullback as kl # modif +Bandits.
import scipy.stats as ss

class FTL:
    def __init__(self,nbArms):
        self.A = nbArms
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)

    def chooseArmToPlay(self):
        return np.argmax(self.Means)

    def receiveReward(self,arm,reward):
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1

    def name(self):
        return "FTL"


class UCB: # TP 2.1
    def __init__(self, nbArms, maxReward=1.):
        self.A = nbArms
        self.Alpha = 3
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
        return "UCB"


class KLUCB:
    def __init__(self, nbArms, maxReward=1.):
        ...
        self.clear()

    def clear(self):
        ...

    def chooseArmToPlay(self):
        return ...

    def receiveReward(self, arm, reward):
        ...

    def name(self):
        return "KL-UCBBern"



class TS:
    def __init__(self, nbArms, maxReward=1.):
        ...
        self.clear()

    def clear(self):
        ...

    def chooseArmToPlay(self):
        return ...

    def receiveReward(self, arm, reward):
        ...

    def name(self):
        return "TS-Bern"