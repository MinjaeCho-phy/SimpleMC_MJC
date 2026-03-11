from simplemc.cosmo.paramDefs import h_par, Ok_par, dft_Oh_par, dft_Oe_par, dft_w_par, dft_l_par
from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.models.DFT1Cosmology import DFT1Cosmology


class DFTCosmology(DFT1Cosmology):
    """
    DFT cosmology with OL=0 (no cosmological constant).
    Free parameters: h, Ok, Oh, Oe, w, l.
    """
    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value, l=dft_l_par.value):
        DFT1Cosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=0.0, Oe=Oe, w=w, l=l)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_Oe_par, dft_w_par, dft_l_par]

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh":
                self.Oh = p.value
            elif p.name == "Oe":
                self.Oe = p.value
            elif p.name == "w_dft":
                self.w = p.value
            elif p.name == "l_dft":
                self.l = p.value
        self.OL = 0.0
        self.initialize()
        return True


class DFTw1l2Cosmology(DFT1Cosmology):
    """
    DFT cosmology with fixed w=1, l=2 and OL=0.
    Free parameters: h, Ok, Oh, Oe.
    """
    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value):
        DFT1Cosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=0.0, Oe=Oe, w=1.0, l=2.0)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_Oe_par]

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh":
                self.Oh = p.value
            elif p.name == "Oe":
                self.Oe = p.value
        self.OL = 0.0
        self.initialize()
        return True


class DFTl3w1Cosmology(DFT1Cosmology):
    """
    DFT cosmology with constraint l = 3w + 1 and OL=0.
    Free parameters: h, Ok, Oh, Oe, w. l is derived from w.
    """
    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value):
        DFT1Cosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=0.0, Oe=Oe, w=w, l=3.0*w+1.0)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_Oe_par, dft_w_par]

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh":
                self.Oh = p.value
            elif p.name == "Oe":
                self.Oe = p.value
            elif p.name == "w_dft":
                self.w = p.value
        self.l = 3.0 * self.w + 1.0
        self.OL = 0.0
        self.initialize()
        return True


class DFTl2wCosmology(DFT1Cosmology):
    """
    DFT cosmology with constraint l = 2w and OL=0.
    Free parameters: h, Ok, Oh, Oe, w. l is derived from w.
    """
    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value):
        DFT1Cosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=0.0, Oe=Oe, w=w, l=2.0*w)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_Oe_par, dft_w_par]

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh":
                self.Oh = p.value
            elif p.name == "Oe":
                self.Oe = p.value
            elif p.name == "w_dft":
                self.w = p.value
        self.l = 2.0 * self.w
        self.OL = 0.0
        self.initialize()
        return True


class DFTl0Cosmology(DFT1Cosmology):
    """
    DFT cosmology with fixed l=0 and OL=0.
    Free parameters: h, Ok, Oh, Oe, w.
    """
    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value):
        DFT1Cosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=0.0, Oe=Oe, w=w, l=0.0)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_Oe_par, dft_w_par]

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh":
                self.Oh = p.value
            elif p.name == "Oe":
                self.Oe = p.value
            elif p.name == "w_dft":
                self.w = p.value
        self.OL = 0.0
        self.l  = 0.0
        self.initialize()
        return True
