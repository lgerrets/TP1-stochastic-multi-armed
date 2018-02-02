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

    def chooseArmToPlay(self): # modified from class UCB
        if self.Time<=self.A: # Draw each arm once
            return self.Time-1
        else:
            mu_tilde = np.zeros(self.A)
            mu_tilde = self.Means+np.sqrt(self.Alpha*np.log(self.Time)/(2*self.NbPulls))
            mu_tilde[mu_tilde>1] = 1
            return np.argmax(mu_tilde)

    def receiveReward(self, arm, reward):
        reward = reward/self.MaxReward
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1
        self.Time += 1

    def name(self):
        return "UCB_3_1-"+str(self.Alpha)

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

    def chooseArmToPlay(self): # modified from class UCB_3_1
        if self.Time<=self.A: # Draw each arm once
            return self.Time-1
        else:
            mu_tilde = np.zeros(self.A)
            mu_tilde = self.Means+np.sqrt(self.Alpha*np.log(self.Time)/(2*self.NbPulls))
            mu_tilde[mu_tilde>1] = 1
            best = np.max(mu_tilde)
            pulls = np.zeros(self.A)
            pulls[mu_tilde!=best] = pulls[0]+42
            return np.argmin(pulls) # = argmin(Na ; a in argmax(min(1,mu_tilde)) )
    
    def receiveReward(self, arm, reward):
        reward = reward/self.MaxReward
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1
        self.Time += 1

    def name(self):
        return "UCB_3_2-"+str(self.Alpha)

class UCB_3_3_a:
    def __init__(self, nbArms, alpha=3, maxReward=1., delta=0.01):
        self.A = nbArms
        if alpha<=1:
            self.Alpha = 3
        else:
            self.Alpha = alpha
        self.Delta = delta
        self.MaxReward = maxReward # les rewards seront normalisées sur [0,1]
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)
        self.Time = 1

    def chooseArmToPlay(self): # modified from class UCB_3_1
        if self.Time<=self.A: # Draw each arm once
            return self.Time-1
        else:
            mu_tilde = np.zeros(self.A)
            mu_tilde = self.Means+np.sqrt(self.Alpha*np.log(np.ceil(np.log(self.Time)/np.log(self.Alpha))/self.Delta)/(2*self.NbPulls))
            return np.argmax(mu_tilde)
    
    def receiveReward(self, arm, reward):
        reward = reward/self.MaxReward
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1
        self.Time += 1

    def name(self):
        return "UCB_3_3_a-"+str(self.Alpha)

class UCB_3_3_b:
    def __init__(self, nbArms, maxReward=1., delta=0.01):
        self.A = nbArms
        self.Delta = delta
        self.MaxReward = maxReward # les rewards seront normalisées sur [0,1]
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)
        self.Time = 1

    def chooseArmToPlay(self): # modified from class UCB_3_1
        if self.Time<=self.A: # Draw each arm once
            return self.Time-1
        else:
            mu_tilde = np.zeros(self.A)
            mu_tilde = self.Means+np.sqrt((1+1/self.NbPulls)*np.log(np.sqrt(self.NbPulls+1)/self.Delta)/(2*self.NbPulls))
            return np.argmax(mu_tilde)
    
    def receiveReward(self, arm, reward):
        reward = reward/self.MaxReward
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1
        self.Time += 1

    def name(self):
        return "UCB_3_3_b"