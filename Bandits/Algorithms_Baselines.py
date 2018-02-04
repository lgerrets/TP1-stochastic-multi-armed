import numpy as np
import Bandits.Algorithms_kullback as kl # modif +Bandits.
import scipy.stats as ss
from sympy.solvers import solve
from sympy import Symbol

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


class KLUCB:
    def __init__(self, nbArms, maxReward=1.):
        self.A = nbArms
        self.MaxReward = maxReward # inutilisé pour le moment
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)
        self.Time = 1

    def chooseArmToPlay(self):
        if self.Time<=self.A: # Draw each arm once
            return self.Time-1
        else:
            mu_max = np.zeros(self.A)
            for a in range(self.A):
                kl_0 = kl.klBern(0,self.Means[a])
                kl_1 = kl.klBern(1,self.Means[a])
                borne = np.log(self.Time)/self.NbPulls[a]
                if kl_0<borne and kl_1<borne:
                    mu_max[a] = max(kl_0,kl,1)
                else:
                    q = Symbol('q')
                    q_sol = solve(kl.klBern(q,self.Means[a])-borne,q)[0] # une recherche dichotomique pourrait être plus optimisée en temps
                    mu_max[a] = self.NbPulls[a]*kl.klBern(q_sol,self.Means[a])
            return np.argmax(mu_max)


    def receiveReward(self, arm, reward):
        reward = reward/self.MaxReward
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1
        self.Time += 1

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