#!/usr/bin/env python3
"""Canonical CASCADE seqlet-calling parameters -- a reasonable, tested default set.

Import these into your own driver script rather than redefining window/flank/FDR
ad hoc, so results stay comparable across runs.
"""

WINDOW = 20      # MoDISco sliding-window smoothing width
FLANK = 5        # flank added each side of the peak window
SEQLET_W = WINDOW + 2 * FLANK        # fixed seqlet width = 30
TARGET_FDR = 0.05    # per-position FDR target for the CASCADE envelope

# MoDISco's extract_seqlets `suppress` is int(0.5*window) + flank
# (non-max suppression radius around each extracted peak). With WINDOW=20,
# FLANK=5 this is int(10) + 5 = 15. See modiscolite.tfmodisco.TFMoDISco.
SUPPRESS = int(0.5 * WINDOW) + FLANK  # = 15

# TFMoDISco's pass-fraction / weak-sign args, passed through to extract_seqlets
# for call-signature compatibility (CASCADE's per-position FDR replaces MoDISco's
# flat-threshold sign refinement, so WEAK_THRESHOLD_FOR_COUNTING_SIGN is unused,
# but the guardrail fractions still apply -- see cascade_seqlets.py).
MIN_PASSING_WINDOWS_FRAC = 0.03
MAX_PASSING_WINDOWS_FRAC = 0.2
WEAK_THRESHOLD_FOR_COUNTING_SIGN = 0.8
