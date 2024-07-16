import math as N
import numpy as np
from simplemc.models.LCDMCosmology import LCDMCosmology
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import C1_par, C2_par, Delta0_par

class CarlevaroCosmology(LCDMCosmology):
    def __init__(self, varyC1=True, varyC2=True, varyDelta0=True):
        self.varyDelta0 = varyDelta0
        self.Delta0 = Delta0_par.value
        self.varyC2 = varyC2
        self.C2 = C2_par.value
        self.varyC1 = varyC1
        self.C1 = C1_par.value


        self.zinter = np.linspace(0.0, 3.0, 50)
        LCDMCosmology.__init__(self, mnu=0)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        if (self.varyDelta0): l.append(Delta0_par)
        if (self.varyC1): l.append(C1_par)
        if (self.varyC2): l.append(C2_par)
        return l

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "C1":
                self.C1 = p.value
            elif p.name == "C2":
                self.C2 = p.value
            elif p.name == "Delta0":
                self.Delta0 = p.value

        self.initialize()
        return True

    
    #def de_dm_rhow(self, func, r):
        #Delta = func[0]
        #F = func[1]
        #dDeltadz = (3/(1+r))*(Delta-self.C1*np.exp(F))
        #dFdz = self.C2/((1+r)*(self.Om*(1+r)**3+(1-self.Om-self.Delta0)+Delta)**0.5)
        #dfuncdz = [dDeltadz, dFdz]
        #return dfuncdz



    #def initialize(self):
    #    rDelta = np.reshape(odeint(self.de_dm_rhow ,[self.Delta0, 0] ,self.zinter)[:,0],len(self.zinter))
        #rF = np.reshape(odeint(self.de_dm_rhow ,[self.Delta0, 0] ,self.zinter)[:,1],len(self.zinter))
    #    self.rDelta_inter = interp1d(self.zinter, rDelta)
        #self.rF_inter = interp1d(self.zinter, rF)
    #    return True

    ## this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    #def RHSquared_a(self,a):
    #    z= 1./a - 1
    #    if self.rDelta_inter(z)>=0:
    #        return (self.Om*(1+z)**3+(1-self.Om-self.Delta0)+self.rDelta_inter(z))**0.5
    #    else:
    #        return (self.Om*(1+z)**3+(1-self.Om-self.Delta0))**0.5


    def de_dm_rhow(self, r, v):
        Delta, F = v
        return [(3/(1+r))*(Delta-self.C1*np.exp(F)), self.C2/((1+r)*(self.Om*(1+r)**3+(1-self.Om-self.C1*(1.0+self.Delta0))+Delta)**0.5)]

    def initialize(self):
        sol = solve_ivp(self.de_dm_rhow, [0.0,3.0], [self.C1*(1.0+self.Delta0), 0.0],dense_output=True)
        self.rDelta_inter = interp1d(self.zinter, sol.sol(self.zinter)[0])
        return True

    ## this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self,a):
        z=1.0/a - 1.0
        if self.rDelta_inter(z)>=0:
            return (self.Om*(1+z)**3+(1-self.Om-self.C1*(1.0+self.Delta0))+self.rDelta_inter(z))
        else:
            return (self.Om*(1+z)**3+(1-self.Om-self.C1*(1.0+self.Delta0)))
 




