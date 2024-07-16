

from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.paramDefs import cd_par, Ok_par


class DonatellaCosmology(LCDMCosmology):
    def __init__(self, varycd=True, varyOk=True):
        """
        This is a cosmology defined by Donatella Fiorucci
        Parameters
        ----------
        

        -------

        """

        self.varycd = varycd
        self.cd = cd_par.value
        self.varyOk = varyOk
        self.Ok = Ok_par.value
        LCDMCosmology.__init__(self)


    # my free parameters. We add w on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        if (self.varycd): l.append(cd_par)
        if (self.varyOk): l.append(Ok_par)
        return l

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "cd":
                self.cd = p.value
            elif p.name == "Ok":
                self.Ok = p.value
                self.setCurvature(self.Ok)
                if (abs(self.Ok) > 1.0):
                    return False
        return True


    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        z= 1./a - 1
        return (self.Ocb/a**3+self.Omrad/a**4+self.Ok/a**2+(1.0-self.Om-self.Ok))/(1-self.cd*(self.Ocb/(1.0+z)+self.Omrad+self.Ok/((1.0+z)**2)+(1.0-self.Om-self.Ok)/((1.0+z)**4)))

