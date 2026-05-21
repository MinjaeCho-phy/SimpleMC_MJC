# coding=utf-8
import sys

# Cosmologies already included
from .models import LCDMCosmology
# Unified DFT/LDFT model: DFTCosmology(varyOL=False) == DFT, varyOL=True == LDFT.
from .models.DFTCosmology import DFTCosmology, DFTw1l2Cosmology, DFTl3w1Cosmology, DFTl2wCosmology, DFTl0Cosmology
from .models.DFTVacuum import DFTVacuum
from .models.DFTMatterRadiation import DFTMatterRadiation
from .models.wCDMCosmology import wCDMCosmology
from .models.owa0CDMCosmology import owa0CDMCosmology

#Generic model
from .models.SimpleModel import SimpleModel, SimpleCosmoModel

# Composite Likelihood
from .likelihoods.CompositeLikelihood import CompositeLikelihood

# Likelihood Multiplier
from .likelihoods.LikelihoodMultiplier import LikelihoodMultiplier

# Likelihood modules
from .likelihoods.BAOLikelihoods import DR11LOWZ, DR11CMASS, DR14LyaAuto, DR14LyaCross, \
                                        SixdFGS, SDSSMGS, DR11LyaAuto, DR11LyaCross, eBOSS, \
                                        DR12Consensus, DR16BAO, DESIBAO
from .likelihoods.SimpleCMBLikelihood import PLK, PLK15, PLK18, WMAP9
from .likelihoods.CompressedSNLikelihood import BetouleSN, UnionSN
from .likelihoods.SNLikelihood import JLASN_Full
from .likelihoods.PantheonSNLikelihood import PantheonSN, BinnedPantheon
from .likelihoods.PantheonPlusSNLikelihood import PantheonPlus
from .likelihoods.PantheonPlusSH0ESLikelihood import PantheonPlusSH0ES
from .likelihoods.FastPantheonLikelihood import FastPantheon
from .likelihoods.UNION3Likelihood import UNION3
from .likelihoods.DESY5Likelihood import DESY5
from .likelihoods.DESDovekieLikelihood import DESDovekie
from .likelihoods.CompressedHDLikelihood import HubbleDiagram, HD23
from .likelihoods.Compressedfs8Likelihood import fs8Diagram
from .likelihoods.HubbleParameterLikelihood import RiessH0, RiessH0_21
from .likelihoods.BBNLikelihood import BBN

from .likelihoods.FineStructureConstantLikelihood import FSC

from .likelihoods.StrongLensingLikelihood import StrongLensing

from .likelihoods.SimpleLikelihood import GenericLikelihood, StraightLine
from .likelihoods.RotationCurvesLikelihood import RotationCurvesLike


# String parser Aux routines
model_list = "LCDM, DFT1"

def ParseModel(model, **kwargs):
    """ 
    Parameters
    -----------
    model:
         name of the model, i.e. LCDM

    Returns
    -----------
    object - info/calculations based on this model: i.e. d_L, d_A, d_H

    """
    custom_parameters = kwargs.pop('custom_parameters', None)
    custom_function = kwargs.pop('custom_function', None)

    if model == "LCDM":
        T = LCDMCosmology()
    elif model == "LCDM_fsc":
        T = LCDMCosmology(fixfsc=False)
    elif model == "wCDM":
        T = wCDMCosmology()
    elif model == "owa0CDM":
        T = owa0CDMCosmology()
    elif model == "wa0CDM":
        # Flat CPL: w0+wa free, Ok forced to 0 (overrides Ok_par.value=1.0
        # DFT-fork default). w0/wa bounds inherited from paramDefs.
        T = owa0CDMCosmology(varyOk=False)
        T.Ok = 0.0
        T.setCurvature(0.0)
    # DFT-family naming convention:
    #   bare name   → fixfsc=False, ishzero=False (alpha_fsc free, Oh free; default)
    #   _fsc suffix → fixfsc=True  (alpha_fsc held at lab value)
    #   _noh suffix → ishzero=True (Omega_h fixed to 0, dropped from free list)
    #   suffixes combine, e.g. _noh_fsc.
    # LDFT* == DFT* with the cosmological constant turned on (varyOL=True).
    elif model == "LDFT":
        T = DFTCosmology(varyOL=True)
    elif model == "LDFT_fsc":
        T = DFTCosmology(varyOL=True, fixfsc=True)
    elif model == "LDFT_noh":
        T = DFTCosmology(varyOL=True, ishzero=True)
    elif model == "LDFT_noh_fsc":
        T = DFTCosmology(varyOL=True, ishzero=True, fixfsc=True)
    elif model == "LDFT_w1l2":
        T = DFTw1l2Cosmology(varyOL=True)
    elif model == "LDFT_w1l2_fsc":
        T = DFTw1l2Cosmology(varyOL=True, fixfsc=True)
    elif model == "LDFT_w1l2_noh":
        T = DFTw1l2Cosmology(varyOL=True, ishzero=True)
    elif model == "LDFT_w1l2_noh_fsc":
        T = DFTw1l2Cosmology(varyOL=True, ishzero=True, fixfsc=True)
    elif model == "LDFT_l3w1":
        T = DFTl3w1Cosmology(varyOL=True)
    elif model == "LDFT_l3w1_fsc":
        T = DFTl3w1Cosmology(varyOL=True, fixfsc=True)
    elif model == "LDFT_l3w1_noh":
        T = DFTl3w1Cosmology(varyOL=True, ishzero=True)
    elif model == "LDFT_l3w1_noh_fsc":
        T = DFTl3w1Cosmology(varyOL=True, ishzero=True, fixfsc=True)
    elif model == "LDFT_l2w":
        T = DFTl2wCosmology(varyOL=True)
    elif model == "LDFT_l2w_fsc":
        T = DFTl2wCosmology(varyOL=True, fixfsc=True)
    elif model == "LDFT_l2w_noh":
        T = DFTl2wCosmology(varyOL=True, ishzero=True)
    elif model == "LDFT_l2w_noh_fsc":
        T = DFTl2wCosmology(varyOL=True, ishzero=True, fixfsc=True)
    elif model == "LDFT_l0":
        T = DFTl0Cosmology(varyOL=True)
    elif model == "LDFT_l0_fsc":
        T = DFTl0Cosmology(varyOL=True, fixfsc=True)
    elif model == "LDFT_l0_noh":
        T = DFTl0Cosmology(varyOL=True, ishzero=True)
    elif model == "LDFT_l0_noh_fsc":
        T = DFTl0Cosmology(varyOL=True, ishzero=True, fixfsc=True)
    elif model == "DFT":
        T = DFTCosmology()
    elif model == "DFT_fsc":
        T = DFTCosmology(fixfsc=True)
    elif model == "DFT_noh":
        T = DFTCosmology(ishzero=True)
    elif model == "DFT_noh_fsc":
        T = DFTCosmology(ishzero=True, fixfsc=True)
    elif model == "DFT_w1l2":
        T = DFTw1l2Cosmology()
    elif model == "DFT_w1l2_fsc":
        T = DFTw1l2Cosmology(fixfsc=True)
    elif model == "DFT_w1l2_noh":
        T = DFTw1l2Cosmology(ishzero=True)
    elif model == "DFT_w1l2_noh_fsc":
        T = DFTw1l2Cosmology(ishzero=True, fixfsc=True)
    elif model == "DFT_l3w1":
        T = DFTl3w1Cosmology()
    elif model == "DFT_l3w1_fsc":
        T = DFTl3w1Cosmology(fixfsc=True)
    elif model == "DFT_l3w1_noh":
        T = DFTl3w1Cosmology(ishzero=True)
    elif model == "DFT_l3w1_noh_fsc":
        T = DFTl3w1Cosmology(ishzero=True, fixfsc=True)
    elif model == "DFT_l2w":
        T = DFTl2wCosmology()
    elif model == "DFT_l2w_fsc":
        T = DFTl2wCosmology(fixfsc=True)
    elif model == "DFT_l2w_noh":
        T = DFTl2wCosmology(ishzero=True)
    elif model == "DFT_l2w_noh_fsc":
        T = DFTl2wCosmology(ishzero=True, fixfsc=True)
    elif model == "DFT_l0":
        T = DFTl0Cosmology()
    elif model == "DFT_l0_fsc":
        T = DFTl0Cosmology(fixfsc=True)
    elif model == "DFT_l0_noh":
        T = DFTl0Cosmology(ishzero=True)
    elif model == "DFT_l0_noh_fsc":
        T = DFTl0Cosmology(ishzero=True, fixfsc=True)
    # DFT2 == l=0 with the cosmological constant on (== LDFT_l0).
    elif model == "DFT2":
        T = DFTl0Cosmology(varyOL=True)
    elif model == "DFTvac":
        T = DFTVacuum()
    elif model == "DFTvac_fsc":
        T = DFTVacuum(fixfsc=True)
    elif model == "DFTvac_noh":
        T = DFTVacuum(ishzero=True)
    elif model == "DFTvac_noh_fsc":
        T = DFTVacuum(ishzero=True, fixfsc=True)
    # Backward-compat aliases — the DFT1Vacuum class has been merged into
    # DFTVacuum, but existing chains under the old names are still reproducible.
    elif model == "DFT1vac":
        T = DFTVacuum()
    elif model == "DFT1vac_fsc":
        T = DFTVacuum(fixfsc=True)
    elif model == "DFT1vac_noh":
        T = DFTVacuum(ishzero=True)
    elif model == "DFT1vac_noh_fsc":
        T = DFTVacuum(ishzero=True, fixfsc=True)
    # DFTmr: "full matter universe" — two critical-line (l=3w-1) species,
    # dust (w=1,l=2) + radiation (w=1/3,l=0), on the Ok+Oh DFT background.
    elif model == "DFTmr":
        T = DFTMatterRadiation()
    elif model == "DFTmr_fsc":
        T = DFTMatterRadiation(fixfsc=True)
    elif model == "DFTmr_noh":
        T = DFTMatterRadiation(ishzero=True)
    elif model == "DFTmr_noh_fsc":
        T = DFTMatterRadiation(ishzero=True, fixfsc=True)
    elif model == 'simple':
        T = SimpleModel(custom_parameters, custom_function)
    elif model == 'simple_cosmo':
        T = SimpleCosmoModel(custom_parameters, RHSquared=custom_function)
    else:
        print("Cannot recognize model", model)
        sys.exit(1)

    return T


data_list = "BBAO, GBAO, GBAO_no6dF, CMASS, LBAO, LaBAO, LxBAO, MGS, Planck, WMAP, PlRd, WRd, PlDa, PlRdx10,"\
    "CMBW, SN, SNx10, UnionSN, RiessH0, 6dFGS, PantheonPlus, DR16BAO, HD23, DESIBAO, FSC, FastPantheon"


def ParseDataset(datasets, **kwargs):
    """ 
    Parameters
    -----------
    datasets:
         name of datasets, i.e. BBAO

    Returns
    -----------
    object - likelihood

    """
    path_to_data = kwargs.pop('path_to_data', None)
    path_to_cov = kwargs.pop('path_to_cov', None)
    fn = kwargs.pop('fn', 'generic')

    dlist = datasets.split('+')
    L = CompositeLikelihood([])
    for name in dlist:
        if name == 'BBAO':
            L.addLikelihoods([
                DR11LOWZ(),
                DR11CMASS(),
                DR11LyaAuto(),
                DR11LyaCross(),
                SixdFGS(),
                SDSSMGS()
            ])
        elif name == 'GBAO11':
            L.addLikelihoods([
                DR11LOWZ(),
                DR11CMASS(),
                SixdFGS(),
                SDSSMGS()
            ])
        elif name == 'CBAO':
            L.addLikelihoods([
                DR12Consensus(),
                DR14LyaAuto(),
                DR14LyaCross(),
                SixdFGS(),
                SDSSMGS(),
                eBOSS()
            ])
        elif name == 'GBAOx10':
            L.addLikelihoods([
                LikelihoodMultiplier(DR11LOWZ(),  100.0),
                LikelihoodMultiplier(DR11CMASS(), 100.0),
                LikelihoodMultiplier(SixdFGS(),   100.0)
            ])
        elif name == 'GBAO_no6dF':
            L.addLikelihoods([
                DR11LOWZ(),
                DR11CMASS()
            ])
        elif name == 'CMASS':
            L.addLikelihoods([
                DR11CMASS()
            ])
        elif name == 'LBAO':
            L.addLikelihoods([
                DR14LyaAuto(),
                DR14LyaCross()
            ])
        elif name == 'LBAO11':
            L.addLikelihoods([
                DR11LyaAuto(),
                DR11LyaCross()
            ])
        elif name == 'LaBAO':
            L.addLikelihoods([
                DR14LyaAuto(),
            ])
        elif name == 'LxBAO':
            L.addLikelihoods([
                DR14LyaCross(),
            ])
        elif name == "MGS":
            L.addLikelihood(SDSSMGS())
        elif name == '6dFGS':
            L.addLikelihood(SixdFGS())
        elif name == 'eBOSS':
            L.addLikelihood(eBOSS())
        elif name == 'DR16BAO':
            L.addLikelihood(DR16BAO())
        elif name == 'DESI':
            L.addLikelihood(DESIBAO())
        elif name == 'PLK':
            L.addLikelihood(PLK())
        elif name == 'PLK15':
            L.addLikelihood(PLK15())
        elif name == 'PLK18':
            L.addLikelihood(PLK18())
        elif name == 'PLKW':
            from .likelihoods.WangWangCMB import PlanckLikelihood
            L.addLikelihood(PlanckLikelihood())
        elif name == 'WMAP9':
            L.addLikelihood(WMAP9())
        elif name == 'PlRd':
            L.addLikelihood(PLK(kill_Da=True))
        elif name == 'WRd':
            L.addLikelihood(WMAP9(kill_Da=True))
        elif name == 'PlDa':
            L.addLikelihood(PLK(kill_rd=True))
        elif name == 'PlRdx10':
            L.addLikelihood(LikelihoodMultiplier(
                PLK(kill_Da=True), 100.0))
        elif name == 'Pantheon':
            L.addLikelihood(PantheonSN())
        elif name == 'BPantheon':
            L.addLikelihood(BinnedPantheon())
        elif name == 'PantheonPlus':
            L.addLikelihood(PantheonPlus())
        elif name == 'JLA':
            L.addLikelihood(JLASN_Full())
        elif name == 'Union3':
            L.addLikelihood(UNION3())
        elif name == 'DESY5':
            L.addLikelihood(DESY5 ())
        elif name == 'SN':
            L.addLikelihood(BetouleSN())
        elif name == 'SNx10':
            L.addLikelihood(LikelihoodMultiplier(BetouleSN(), 100.0))
        elif name == 'UnionSN':
            L.addLikelihood(UnionSN())
        elif name == 'RiessH0':
            L.addLikelihood(RiessH0())
        elif name == 'RiessH0_21':
            L.addLikelihood(RiessH0_21())
        elif name == 'HD':
            L.addLikelihood(HubbleDiagram())
        elif name == 'HD23':
            L.addLikelihood(HD23())
        elif name == 'BBN':
            L.addLikelihood(BBN())
        elif name == 'fs8':
            L.addLikelihood(fs8Diagram())
        elif name == 'dline':
            L.addLikelihood(StraightLine())
        #elif name == 'CPantheon_15':
        #    L.addLikelihood(PantheonLikelihood())
        elif name == 'RC':
            L.addLikelihood(RotationCurvesLike())
        elif name == "SL":
            L.addLikelihood(StrongLensing())
        elif name == 'FSC':
            L.addLikelihood(FSC())
        elif name == 'FastPantheon':
            L.addLikelihood(FastPantheon())
        elif name == 'PantheonPlusSH0ES':
            L.addLikelihood(PantheonPlusSH0ES())
        elif name == 'DESDovekie':
            L.addLikelihood(DESDovekie())
        elif name == 'generic':
            L.addLikelihood(GenericLikelihood(path_to_data=path_to_data,
                                              path_to_cov=path_to_cov,
                                              fn=fn))
        else:
            print("Cannot parse data, unrecognizable part:", name)
            sys.exit(1)

    return L
