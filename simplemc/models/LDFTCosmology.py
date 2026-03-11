from simplemc.cosmo.paramDefs import h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par, dft_l_par
from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.models.DFT1Cosmology import DFT1Cosmology


class LDFTCosmology(DFT1Cosmology):
    """
    DFT1 cosmology with OL (cosmological constant) as a free parameter.
    All parameters free: h, Ok, Oh, OL, Oe, w, l.
    Alias for DFT1Cosmology with explicit LDFT naming convention.
    """
    pass


class LDFTw1l2Cosmology(DFT1Cosmology):
    """
    LDFT cosmology with fixed w=1, l=2. OL is a free parameter.
    Free parameters: h, Ok, Oh, OL, Oe.
    """
    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value):
        DFT1Cosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe, w=1.0, l=2.0)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par]

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh":
                self.Oh = p.value
            elif p.name == "OL":
                self.OL = p.value
            elif p.name == "Oe":
                self.Oe = p.value
        self.initialize()
        return True


class LDFTl3w1Cosmology(DFT1Cosmology):
    """
    LDFT cosmology with constraint l = 3w + 1. OL is a free parameter.
    Free parameters: h, Ok, Oh, OL, Oe, w. l is derived from w.
    """
    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value, w=dft_w_par.value):
        DFT1Cosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe, w=w, l=3.0*w+1.0)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par]

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh":
                self.Oh = p.value
            elif p.name == "OL":
                self.OL = p.value
            elif p.name == "Oe":
                self.Oe = p.value
            elif p.name == "w_dft":
                self.w = p.value
        self.l = 3.0 * self.w + 1.0
        self.initialize()
        return True


class LDFTl2wCosmology(DFT1Cosmology):
    """
    LDFT cosmology with constraint l = 2w. OL is a free parameter.
    Free parameters: h, Ok, Oh, OL, Oe, w. l is derived from w.
    """
    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value, w=dft_w_par.value):
        DFT1Cosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe, w=w, l=2.0*w)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par]

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh":
                self.Oh = p.value
            elif p.name == "OL":
                self.OL = p.value
            elif p.name == "Oe":
                self.Oe = p.value
            elif p.name == "w_dft":
                self.w = p.value
        self.l = 2.0 * self.w
        self.initialize()
        return True


class LDFTl0Cosmology(DFT1Cosmology):
    """
    LDFT cosmology with fixed l=0. OL is a free parameter.
    Free parameters: h, Ok, Oh, OL, Oe, w.
    """
    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value, w=dft_w_par.value):
        DFT1Cosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe, w=w, l=0.0)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par]

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh":
                self.Oh = p.value
            elif p.name == "OL":
                self.OL = p.value
            elif p.name == "Oe":
                self.Oe = p.value
            elif p.name == "w_dft":
                self.w = p.value
        self.l = 0.0
        self.initialize()
        return True
