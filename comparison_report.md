# Comparison Report: Local vs Escamilla

**Date:** 2026-01-16
**Comparison:** Local `test` branch vs `escamilla/master`
**Focus:** `simplemc` package analysis code (excluding `chains`)

## Summary
The Local repository appears to be a **highly specialized fork** of the SimpleMC framework, stripped of most general purpose models and enhanced with custom "DFT" cosmology models and datasets. The `escamilla` repository represents the comprehensive, "full" version of the library with dozens of models not present locally.

## 1. Models (`simplemc/models/`)
This is the most significant difference.

### Local Only (Custom Additions)
These models exist **only** in your local version:
-   `DFT1Cosmology.py`
-   `DFT2Cosmology.py`

### Escamilla Only (Missing Locally)
Your local version is missing a vast number of models present in `escamilla`, including:
-   `JordiCDMCosmology.py`
-   `EarlyDECosmology.py`
-   `QuintomCosmology.py`
-   `BarrowHDECosmology.py`
-   `PolyCDMCosmology.py`
-   ...and approx. 40 others.

**Key Difference in `ParseModel`:**
The `ParseModel` function in `simplemc/models/__init__.py` confirms this split.
-   **Local**: Supports only `LCDM`, `DFT1`, `DFT2`.
-   **Escamilla**: Supports `LCDMmasslessnu`, `nuLCDM`, `wCDM`, `CPL`, `JordiCDM`, `EarlyDE`, etc.

## 2. Parameter Definitions (`simplemc/cosmo/paramDefs.py`)
-   **Local**: Contains the localized change fixing `Ok` (curvature) to ~1.0.
-   **Escamilla**: Uses the standard broader range for `Ok`.

## 3. Datasets (`simplemc/data/` & `simplemc/likelihoods/`)
Your local version has custom datasets and likelihoods tailored for your analysis.

### Local Only
-   **Datasets**: `Minjae, FSC` (likely in `simplemc/data/`)
-   **Likelihoods**:
    -   `FineStructureConstantLikelihood.py` (FSC)
    -   `PantheonSNFixedLikelihood.py`
-   **Registered in `ParseDataset`**: `Minjae`, `FSC`, `Pantheon_fixed`.

### Escamilla Only
-   Standard datasets not present or enabled in your stripped version (though the diff shows broad overlap in standard files, the configuration logic differs).

## 4. Execution Scripts
-   **Local**: Uses `test1.py` and `run_remaining.py` for execution.
-   **Escamilla**: Relies on the standard `DriverMC.py` workflow without these custom drivers.
-   **DriverMC.py**: This file is **identical** in both the local and `escamilla` repositories. There are no changes to the main driver logic itself, only how it is invoked (via custom scripts locally).

## Conclusion
You are working on a **DFT-specific branch** that has removed most of the "clutter" of other exotic cosmological models from the original library to focus specifically on DFT1/DFT2 models and specific datasets (Minjae, FSC). The `escamilla` repo is the "kitchen sink" version with all available models.
