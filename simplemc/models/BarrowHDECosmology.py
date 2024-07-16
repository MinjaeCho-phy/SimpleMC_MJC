from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import Ok_par
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy import optimize
import numpy as np



class BarrowHDECosmology(LCDMCosmology):
    """
        This is Holographic Dark Energy in modfied Barrow Cosmology.
        This class inherits LCDMCosmology class as the rest of the cosmological
        models already included in SimpleMC.

        :param varyc: variable w0 parameter

    """
    def __init__(self, varyOk=False, varyc=True,varyb=True):
        # Holographic and Barrow parameter
        self.c_par = Parameter("c", 0.7, 0.05, (0.5, 1.5), "c")
        self.b_par = Parameter("b", 1.0, 0.05, (0.0,2.5),  "b") 
        self.varyOk = varyOk
        self.varyc  = varyc
        self.varyb  = varyb  
        
        self.Ok = Ok_par.value
        self.c  = self.c_par.value # holographic parameter 
        self.b  = self.b_par.value # barrow parameter

        # This value is quite related to the initial z
        self.zini = 3
        self.xini = np.log(1./(1+self.zini))
        #self.scale = 10**(-2)
        self.xvals = np.linspace(0,self.xini, 50)

        LCDMCosmology.__init__(self)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        if (self.varyOk): l.append(Ok_par)
        if (self.varyc):  l.append(self.c_par)
        if (self.varyb):  l.append(self.b_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name   == "c":
                self.c = p.value
            elif p.name == "b":
                self.b = p.value
            elif p.name == "Ok":
                  self.Ok = p.value
                  self.setCurvature(self.Ok)
                  if (abs(self.Ok) > 1.0):
                      return False

        self.initialize()
        return True

# x = ln(a)
    def RHS_x_hde_barrow(self, vals,x):
        Omega = vals
        dOmega = -Omega*(Omega - 1)*(1 + self.b + (2 - self.b)*((Omega/self.c**2)**(1./(2*self.b))))
        #print(self.c)
        return dOmega



    #$def compute_Ode(self, Ode_ini):
      #  Ode = Ode_ini*self.scale
       # solution = odeint(self.RHS_a_hde_barrow, Ode, self.avals, h0=1E-5)
        #return solution


    #def ini_sol(self, Ode_ini):
     #   diference = self.compute_Ode(Ode_ini)[-1] - (1-self.Om-self.Ok)
      #  return diference


    def initialize(self):
        """
        Main method that searches the initial conditions for a given model.
        """
             #ini_val = optimize.newton(self.ini_sol, 1)
        Ode0 = (1 - self.Om - self.Ok)
        #Ode = self.compute_Ode(ini_val)

        #ini_vals = Ode0
        result_E = odeint(self.RHS_x_hde_barrow, Ode0, self.xvals)
        #Ode = np.exp(result_E[:, 0])
        self.Ode = interp1d(self.xvals, result_E[:, 0])

        ## Add a flag in case the ini condition isn't found, i.e. c<0.4
        return True



    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        x = np.log(a)
        hubble = (self.Omrad/a**4 + self.Om/a**3)/(1-self.Ode(x))
        #print(hubble)
        return hubble

