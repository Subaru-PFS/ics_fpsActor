#!/usr/bin/env python
"""Extended fps regression simulator — deep end-to-end edition.

Architecture:
  - PhysicsFPGAProtocol (pfiSimServer.py) runs as an asyncio thread on localhost:4001
  - Real CobraCoach connects over TCP to it
  - cc.exposeAndExtractPositions is monkey-patched (MCS camera replacement)
  - Every production code path (engineer.py, pfi.py, func.RUN, ethernet.sock) runs unmodified

Covers:
  createHomeDesign, createBlackDotDesign, createDotConvergenceDesign,
  createThetaPhiScanDesign, cobraMoveAngles, cobraMoveSteps,
  moveToHome, moveToSafePosition, moveToPfsDesign,
  genPfsConfigFromMcs, makeMotorMapGroups, testLoop
  + the original dot-convergence end-to-end test.

Run:
    python sim_dot_convergence.py
"""
import logging
import os
import sys
import datetime
import pathlib
import tempfile
import time as _time

import numpy as np
import pandas as pd

from ics.fpsActor.Commands.FpsCmd import FpsCmd
from ics.cobraCharmer.cobraCoach.cobraCoach import CobraCoach
from ics.cobraCharmer.cobraCoach import engineer as eng
import ics.fpsActor.utils.pfsDesign as pfsDesignUtils
import ics.fpsActor.utils.pfsConfig as pfsConfigUtils
from pfs.utils.database import opdb
from pfs.datamodel import TargetType, FiberStatus, PfsConfig, TargetValidation, CobraCommand
from pfs.utils.fiberids import FiberIds

# Dot handling shipped with INSTRM-2845 (now merged) as utils/dotGeometry.  The guard used
# to import utils/dotConvergence.DotConverger, which that merge removed, so the whole dot
# test skipped silently and the geometry went unchecked.
try:
    from ics.fpsActor.utils.alfUtils import sgfm
    from ics.fpsActor.utils import dotGeometry  # noqa: F401
    HAS_DOT_CONVERGENCE = True
except (ImportError, AttributeError):
    HAS_DOT_CONVERGENCE = False


DUMP_DIR = os.path.expanduser('~/tmp/claude/')
os.makedirs(DUMP_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fake command infrastructure (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class FakeKeyword:
    def __init__(self, value):
        self.values = [value]


class FakeKeywords(dict):
    pass


class FakeCmdInner:
    def __init__(self, keywords):
        converted = {}
        for k, v in keywords.items():
            converted[k] = v if isinstance(v, FakeKeyword) else FakeKeyword(v)
        self.keywords = FakeKeywords(converted)


class FakeCmd:
    def __init__(self, keywords=None):
        self.cmd = FakeCmdInner(keywords or {})

    def inform(self, msg):
        text = msg
        if text.startswith('text="') and text.endswith('"'):
            text = text[6:-1]
        print(f'  [inform] {text}')

    def warn(self, msg):
        print(f'  [WARN]   {msg}')

    def fail(self, msg):
        raise RuntimeError(msg)

    def finish(self, msg=''):
        text = msg
        if text.startswith('text="') and text.endswith('"'):
            text = text[6:-1]
        print(f'  [finish] {text}')

    def isAlive(self):
        return True


class FakeCmdVar:
    didFail = False


class FakeCmdr:
    def call(self, actor, cmdStr, forUserCmd=None, timeLim=60):
        print(f'  [cmdr→{actor}] {cmdStr}')
        return FakeCmdVar()


class FakeVisitor:
    def __init__(self, db):
        self._db = db
        self.frameSeq = 0

    def setOrGetVisit(self, cmd):
        visitId = allocateSimVisitId(self._db)
        self.frameSeq = 0
        return visitId

    def getNextFrameNum(self):
        self.frameSeq += 1
        return self.frameSeq


class FakeCam:
    """Minimal camera stub so cc.cam.filePrefix/resetStack checks don't crash."""
    filePrefix = 'SIM'

    def resetStack(self, *args, **kwargs):
        pass


class FakeActor:
    def __init__(self, db):
        self.visitor    = FakeVisitor(db)
        self.cmdr       = FakeCmdr()
        self.bcast      = FakeCmd()
        self.actorConfig = {}
        self.models     = {}


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def allocateSimVisitId(db):
    """Allocate a fresh visit ID and register it in pfs_visit."""
    maxVisit = db.query_scalar('SELECT COALESCE(MAX(pfs_visit_id), 0) FROM pfs_visit')
    visitId  = int(maxVisit) + 1
    db.insert_kw('pfs_visit',
                 pfs_visit_id=visitId,
                 pfs_visit_description='sim_dot_convergence',
                 issued_at=datetime.datetime.utcnow().isoformat())
    return visitId


def insertSimMcsFrame(db, visitId, iteration, positions, dotPos, rDot, centers):
    """Insert synthetic cobra_target + mcs_exposure + mcs_data + cobra_match rows."""
    nCobras  = len(positions)
    frameId  = visitId * 100 + iteration
    underDot = np.abs(positions - dotPos) < rDot

    if iteration == 0:
        ctRows = pd.DataFrame({
            'pfs_visit_id':      [visitId]   * nCobras,
            'iteration':         [iteration] * nCobras,
            'cobra_id':          list(range(1, nCobras + 1)),
            'pfi_nominal_x_mm':  centers.real.tolist(),
            'pfi_nominal_y_mm':  centers.imag.tolist(),
            'pfi_target_x_mm':   positions.real.tolist(),
            'pfi_target_y_mm':   positions.imag.tolist(),
        })
        db.insert_dataframe('cobra_target', ctRows)

    db.insert_kw('mcs_exposure', mcs_frame_id=frameId, pfs_visit_id=visitId)

    visibleIdx = np.where(~underDot)[0]
    if len(visibleIdx):
        mcsRows = pd.DataFrame({
            'mcs_frame_id':      [frameId] * len(visibleIdx),
            'spot_id':           visibleIdx.tolist(),
            'mcs_center_x_pix':  positions[visibleIdx].real * 100,
            'mcs_center_y_pix':  positions[visibleIdx].imag * 100,
        })
        db.insert_dataframe('mcs_data', mcsRows)

    spotIds  = np.where(underDot, -1, np.arange(nCobras)).tolist()
    frameIds = [None if u else frameId for u in underDot]
    matchRows = pd.DataFrame({
        'pfs_visit_id':    [visitId]   * nCobras,
        'iteration':       [iteration] * nCobras,
        'cobra_id':        list(range(1, nCobras + 1)),
        'mcs_frame_id':    frameIds,
        'spot_id':         spotIds,
        'pfi_center_x_mm': positions.real.tolist(),
        'pfi_center_y_mm': positions.imag.tolist(),
    })
    db.insert_dataframe('cobra_match', matchRows)



def insertSimCobraMatch(db, visitId, iteration, targets, measured, detected):
    """Insert cobra_target + cobra_match (+mcs rows) for a measured/detected pattern.

    Unlike insertSimMcsFrame this does not assume the dot geometry decides visibility --
    the caller supplies `detected`, so a test can inject convergence failures and MCS
    losses independently.  cobra_target must come first: cobra_match carries a foreign
    key on (pfs_visit_id, iteration, cobra_id).
    """
    nCobras = len(measured)
    frameId = visitId * 100 + iteration

    db.insert_dataframe('cobra_target', pd.DataFrame({
        'pfs_visit_id':     [visitId]   * nCobras,
        'iteration':        [iteration] * nCobras,
        'cobra_id':         list(range(1, nCobras + 1)),
        'pfi_nominal_x_mm': targets.real.tolist(),
        'pfi_nominal_y_mm': targets.imag.tolist(),
        'pfi_target_x_mm':  targets.real.tolist(),
        'pfi_target_y_mm':  targets.imag.tolist(),
    }))
    db.insert_kw('mcs_exposure', mcs_frame_id=frameId, pfs_visit_id=visitId)

    seen = np.where(detected)[0]
    if len(seen):
        db.insert_dataframe('mcs_data', pd.DataFrame({
            'mcs_frame_id':     [frameId] * len(seen),
            'spot_id':          seen.tolist(),
            'mcs_center_x_pix': measured[seen].real * 100,
            'mcs_center_y_pix': measured[seen].imag * 100,
        }))

    db.insert_dataframe('cobra_match', pd.DataFrame({
        'pfs_visit_id':    [visitId]   * nCobras,
        'iteration':       [iteration] * nCobras,
        'cobra_id':        list(range(1, nCobras + 1)),
        'mcs_frame_id':    [frameId if d else None for d in detected],
        'spot_id':         np.where(detected, np.arange(nCobras), -1).tolist(),
        'pfi_center_x_mm': measured.real.tolist(),
        'pfi_center_y_mm': measured.imag.tolist(),
    }))


# ─────────────────────────────────────────────────────────────────────────────
# FakeFpsCmd factory
# ─────────────────────────────────────────────────────────────────────────────

def makeFakeFpsCmd(cc, db):
    """Build a minimal FpsCmd instance using the real CobraCoach.

    FpsCmd.cc is a property backed by self.actor.cc, so actor must be set
    before cc — that's why we assign fps.actor first then fps.actor.cc.
    """
    fps              = FpsCmd.__new__(FpsCmd)
    fps.actor        = FakeActor(db)
    fps.actor.cc     = cc
    fps.atThetas     = cc.cobraInfo['thetaAngle'].copy()
    fps.atPhis       = cc.cobraInfo['phiAngle'].copy()
    fps.logger       = logging.getLogger('simFpsCmd')
    fps.xml          = None
    fps.nv           = None
    fps.tranMatrix   = None
    fps.dotConverger = None
    fps._collectVersions = lambda: {}
    return fps


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resetToHome(cc, physics):
    """Teleport truth state and cc.cobraInfo to theta/phi home."""
    nAll      = len(cc.allCobras)
    thetaHome = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (2*np.pi) + np.pi)
    with physics.lock:
        physics.trueThetas[:] = thetaHome
        physics.truePhis[:]   = 0.0
    cc.cobraInfo['thetaAngle'] = thetaHome.copy()
    cc.cobraInfo['phiAngle']   = np.zeros(nAll)
    cc.cobraInfo['position']   = cc.pfi.anglesToPositions(cc.allCobras, thetaHome, np.zeros(nAll))


def _makeHomeDesign(cc):
    """Create a home pfsDesign (ENGINEERING targets at home angles) and write it."""
    nAll      = len(cc.allCobras)
    thetaHome = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (2*np.pi) + np.pi)
    phiHome   = np.zeros(nAll)
    positions = cc.pfi.anglesToPositions(cc.allCobras, thetaHome, phiHome)
    xy        = np.vstack((positions.real, positions.imag)).T
    pfsDesign = pfsDesignUtils.createPfsDesign(cc.calibModel, xy,
                                               moveTargetType=TargetType.ENGINEERING,
                                               designName='sim_home_test')
    pfsDesignUtils.writeDesign(pfsDesign)
    return pfsDesign


def _makeScienceDesign(cc):
    """Create a SCIENCE pfsDesign at in-patrol targets and write it.

    moveToPfsDesign only commands SCIENCE/SKY/FLUXSTD/UNASSIGNED/BLACKSPOT; a HOME design
    belongs to moveToHome.  Targets are built from valid angles so they are reachable, and
    sit away from home so the designId cannot collide with the home design -- writeDesign
    does not overwrite, so a collision would silently feed back the other design.
    """
    nAll      = len(cc.allCobras)
    thetaHome = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (2*np.pi) + np.pi)
    targetT   = thetaHome * 0.5
    targetP   = np.full(nAll, np.deg2rad(60.0))
    positions = cc.pfi.anglesToPositions(cc.allCobras, targetT, targetP)
    xy        = np.vstack((positions.real, positions.imag)).T
    pfsDesign = pfsDesignUtils.createPfsDesign(cc.calibModel, xy,
                                               moveTargetType=TargetType.SCIENCE,
                                               designName='sim_science_test')
    pfsDesignUtils.writeDesign(pfsDesign)
    return pfsDesign


# ─────────────────────────────────────────────────────────────────────────────
# Per-command test functions
# ─────────────────────────────────────────────────────────────────────────────

def test_createDesigns(cc, physics, db):
    """createHomeDesign, createBlackDotDesign, createThetaPhiScanDesign,
    createDotConvergenceDesign."""
    print('\n── test_createDesigns ─────────────────────────────────')
    _resetToHome(cc, physics)
    fps = makeFakeFpsCmd(cc, db)

    pfsDesign = fps._createHomeDesign(FakeCmd())
    assert pfsDesign is not None, 'createHomeDesign returned None'
    assert pfsDesign.pfsDesignId != 0
    print(f'  createHomeDesign OK: 0x{pfsDesign.pfsDesignId:016x}')

    fps.createBlackDotDesign(FakeCmd())
    print('  createBlackDotDesign OK')

    fps.createThetaPhiScanDesign(FakeCmd({'thetaAngle': FakeKeyword(60),
                                          'phiAngle':   FakeKeyword(30)}))
    print('  createThetaPhiScanDesign OK')

    if HAS_DOT_CONVERGENCE:
        fps.createDotConvergenceDesign(FakeCmd())
        print('  createDotConvergenceDesign OK')
    else:
        print('  createDotConvergenceDesign SKIP (not on this branch)')

    print('  PASS')
    return True


def test_cobraMoveAngles(cc, physics, db):
    """cobraMoveAngles phi +10° — verify cc.cobraInfo updated via full stack.

    Starting from phi=20° (not phi=0) to avoid the positionsToAngles solution-0 artefact
    for the 14 cobras with phiIn < -π where solution-0 is wrong below ~17.2°.
    """
    print('\n── test_cobraMoveAngles ───────────────────────────────')
    _resetToHome(cc, physics)

    # Start from phi=20° so every cobra is above the positionsToAngles ambiguity threshold
    nAll      = len(cc.allCobras)
    phiStart  = np.full(nAll, np.deg2rad(20.0))
    with physics.lock:
        physics.truePhis[:] = phiStart
    cc.cobraInfo['phiAngle'] = phiStart.copy()
    cc.cobraInfo['position'] = cc.pfi.anglesToPositions(
        cc.allCobras, cc.cobraInfo['thetaAngle'], phiStart)

    fps     = makeFakeFpsCmd(cc, db)
    goodIdx = cc.goodIdx
    phiBefore = cc.cobraInfo['phiAngle'][goodIdx].copy()

    fps.cobraMoveAngles(FakeCmd({'phi': FakeKeyword(True), 'angle': FakeKeyword(10.0)}))

    phiAfter = cc.cobraInfo['phiAngle'][goodIdx]
    expected = np.deg2rad(10.0)
    err = np.max(np.abs(phiAfter - phiBefore - expected))
    assert err < np.deg2rad(0.5), f'phi not updated correctly: max error = {np.rad2deg(err):.3f}°'
    print(f'  phi +10° from 20° start OK (max error {np.rad2deg(err):.3f}°)')
    print('  PASS')
    return True


def test_cobraMoveSteps(cc, physics, db):
    """cobraMoveSteps phi 100 steps — verify physics.truePhis changed."""
    print('\n── test_cobraMoveSteps ────────────────────────────────')
    _resetToHome(cc, physics)
    with physics.lock:
        phiBefore = physics.truePhis[cc.goodIdx].copy()

    fps = makeFakeFpsCmd(cc, db)
    fps.cobraMoveStepsCmd(FakeCmd({'phi': FakeKeyword(True), 'stepsize': FakeKeyword(100)}))

    with physics.lock:
        phiAfter = physics.truePhis[cc.goodIdx].copy()
    moved = np.sum(phiAfter != phiBefore)
    assert moved > 0, 'physics.truePhis unchanged after cobraMoveSteps'
    print(f'  {moved} cobras moved in phi ✓')
    print('  PASS')
    return True


def test_moveToHome(cc, physics, db):
    """moveToHome all arms — verify cc.cobraInfo at home angles."""
    print('\n── test_moveToHome ────────────────────────────────────')
    thetaHome = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (2*np.pi) + np.pi)
    goodIdx   = cc.goodIdx

    # Start from a non-home position
    with physics.lock:
        physics.trueThetas[:] = thetaHome * 0.5
        physics.truePhis[:]   = np.deg2rad(45)
    cc.cobraInfo['thetaAngle'] = thetaHome * 0.5
    cc.cobraInfo['phiAngle']   = np.full(len(cc.allCobras), np.deg2rad(45))
    cc.cobraInfo['position']   = cc.pfi.anglesToPositions(
        cc.allCobras, cc.cobraInfo['thetaAngle'], cc.cobraInfo['phiAngle'])

    fps = makeFakeFpsCmd(cc, db)
    fps._finalizeWriteIngestPfsConfig = lambda *a, **kw: None
    fps.getPfsConfig = lambda cmd, visit, pfsDesign, maskFile=None: \
        pfsConfigUtils.pfsConfigFromDesign(pfsDesign, visit, calibModel=cc.calibModel)

    fps.moveToHome(FakeCmd({'noMCSexposure': FakeKeyword(True)}))

    # noMCS=True path: cc.moveToHome calls cc.setCurrentAngles directly
    assert np.allclose(cc.cobraInfo['phiAngle'][goodIdx],   0.0,               atol=1e-6), \
        'phi not at home'
    assert np.allclose(cc.cobraInfo['thetaAngle'][goodIdx], thetaHome[goodIdx], atol=1e-6), \
        'theta not at home'
    print(f'  thetaAngle ≈ thetaHome ✓   phiAngle ≈ 0 ✓')
    print('  PASS')
    return True


def _degenMotorMapMask(cc, maxSteps=2000, minDegPerBurst=1.0, targetT=None, maxTries=6):
    """Return boolean mask (len=nAll) True for cobras that cannot converge due to bad motor maps.

    Two failure modes:
      - Too fine: maxSteps moves < minDegPerBurst degrees (calibration corruption).
      - Too slow (with targetT supplied): travel from thetaHome to target exceeds
        maxTries bursts using the slower of CCW/CW maps.  maxTries=6 gives a 2-iteration
        margin against the 8-iteration command budget, catching borderline-slow cobras
        (e.g. cobra 454 with CW=31.8°/2k) that empirically fail the 100µm threshold.
    """
    cm   = cc.calibModel
    nAll = len(cc.allCobras)
    mask = np.zeros(nAll, dtype=bool)
    thetaHome = ((cm.tht1 - cm.tht0 + np.pi) % (2*np.pi) + np.pi)

    for cId in range(nAll):
        angArr  = cm.thtOffsets[cId]
        ang_deg = np.rad2deg(angArr[-1])
        dpb_cw  = maxSteps / (cm.negThtSlowSteps[cId, -1] / ang_deg) \
                  if cm.negThtSlowSteps[cId, -1] > 0 else 999.0
        dpb_ccw = maxSteps / (cm.posThtSlowSteps[cId, -1] / ang_deg) \
                  if cm.posThtSlowSteps[cId, -1] > 0 else 999.0
        deg_per_burst = min(dpb_cw, dpb_ccw)
        if deg_per_burst < minDegPerBurst:
            mask[cId] = True
            continue
        if targetT is not None and deg_per_burst > 0:
            travel_deg = np.rad2deg(abs(thetaHome[cId] - targetT[cId]))
            if travel_deg / deg_per_burst > maxTries:
                mask[cId] = True
    return mask


def test_moveToSafePosition(cc, physics, db):
    """moveToSafePosition — verify cc.cobraInfo updated to ~60°/80° via full stack.

    Cobras with degenerate motor maps (2000 steps < 1°) are identified and excluded
    from the max-error assertion — they cannot converge in practice either.
    """
    print('\n── test_moveToSafePosition ────────────────────────────')
    _resetToHome(cc, physics)
    fps = makeFakeFpsCmd(cc, db)

    fps.moveToSafePosition(FakeCmd({'noHome': FakeKeyword(True)}))

    goodIdx        = cc.goodIdx
    cm             = cc.calibModel
    thetaMarginVal = np.deg2rad(15.0)
    nAll           = len(cc.allCobras)

    # Reconstruct the local-angle targets that moveThetaPhi aimed for (local=False path).
    # theta_global=60° is converted to local, then margin-corrected.
    targetT = (np.deg2rad(60.0) - cm.tht0) % (2*np.pi)
    targetT[targetT < thetaMarginVal] += 2*np.pi
    thetaRange = (cm.tht1 - cm.tht0 + np.pi) % (2*np.pi) + np.pi
    targetT[targetT > thetaRange - thetaMarginVal] = (thetaRange - thetaMarginVal)[
        targetT > thetaRange - thetaMarginVal]
    # phi_global=80° to local: phis - phiIn - π (same formula in moveThetaPhi)
    targetP = np.deg2rad(80.0) - cm.phiIn - np.pi

    # Compute the target PFI positions for all cobras
    targetPos = cc.pfi.anglesToPositions(cc.allCobras, targetT, targetP)

    # Exclude cobras that cannot converge: too-fine motor maps OR travel > 8 bursts
    degenMask = _degenMotorMapMask(cc, targetT=targetT)
    checkIdx  = goodIdx[~degenMask[goodIdx]]
    nDegen    = np.sum(degenMask[goodIdx])
    if nDegen:
        print(f'  Excluding {nDegen} cobra(s) with degenerate motor maps '
              f'({len(checkIdx)}/{len(goodIdx)} cobras checked)')

    # Assert position convergence: actual PFI pos vs target PFI pos
    posErrs = np.abs(cc.cobraInfo['position'][checkIdx] - targetPos[checkIdx])
    errPos  = np.max(posErrs)
    assert errPos < 0.1, f'safe-pos not reached (max position err={errPos:.3f} mm)'
    print(f'  converged to (global theta=60°, phi=80°) within {errPos:.3f} mm ✓  '
          f'({len(checkIdx)}/{len(goodIdx)} cobras checked)')
    print('  PASS')
    return True


def test_moveToPfsDesign(cc, physics, db):
    """moveToPfsDesign — real convergence loop through full stack."""
    print('\n── test_moveToPfsDesign ───────────────────────────────')
    thetaHome = ((cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (2*np.pi) + np.pi)

    pfsDesign = _makeScienceDesign(cc)
    designId  = pfsDesign.pfsDesignId

    # Start from a slightly off-home position
    with physics.lock:
        physics.trueThetas[:] = thetaHome * 0.8
        physics.truePhis[:]   = np.deg2rad(20)
    cc.cobraInfo['thetaAngle'] = thetaHome * 0.8
    cc.cobraInfo['phiAngle']   = np.full(len(cc.allCobras), np.deg2rad(20))
    cc.cobraInfo['position']   = cc.pfi.anglesToPositions(
        cc.allCobras, cc.cobraInfo['thetaAngle'], cc.cobraInfo['phiAngle'])

    fps = makeFakeFpsCmd(cc, db)
    fps.getPfsConfig = lambda cmd, visit, pfsDesign, maskFile=None: \
        pfsConfigUtils.pfsConfigFromDesign(pfsDesign, visit, calibModel=cc.calibModel)
    fps._finalizeWriteIngestPfsConfig = lambda *a, **kw: None
    fps.loadModel = lambda *a, **kw: None

    cmd = FakeCmd({
        'designId':                     FakeKeyword(designId),
        'twoStepsOff':                  FakeKeyword(True),
        'noTweak':                      FakeKeyword(True),
        'goHome':                       FakeKeyword(True),
        'skipFiducialInterferenceCheck': FakeKeyword(True),
    })
    fps.moveToPfsDesign(cmd)

    assert fps.atThetas is not None, 'fps.atThetas is None after moveToPfsDesign'
    assert fps.atPhis   is not None, 'fps.atPhis is None after moveToPfsDesign'
    goodIdx = cc.goodIdx
    print(f'  atThetas mean={np.rad2deg(fps.atThetas[goodIdx]).mean():.1f}°  '
          f'atPhis mean={np.rad2deg(fps.atPhis[goodIdx]).mean():.1f}°')

    visit = fps.actor.visitor._db.query_scalar(
        'SELECT MAX(pfs_visit_id) FROM cobra_target')
    ct_count = int(fps.actor.visitor._db.query_dataframe(
        f'SELECT count(*) FROM cobra_target WHERE pfs_visit_id = {visit}').squeeze())
    assert ct_count > 0, 'no cobra_target rows written'
    print(f'  cobra_target rows: {ct_count} for visit {visit} ✓')
    print('  PASS')
    return True


def test_finalizeFiberStatus(cc, physics, db):
    """finalize() — fiberStatus from a realistic cobra_match distribution.

    moveToPfsDesign stubs _finalizeWriteIngestPfsConfig, so nothing else in this suite
    executes finalize(): no cobra_match query, no pfiCenter update, no fiberStatus at all.
    This drives it directly with an injected convergence distribution -- a bulk inside
    tolerance, a tail beyond it, and a fraction MCS never sees -- and checks each
    population lands on the status the rules say it should.
    """
    print('\n── test_finalizeFiberStatus ───────────────────────────')
    TOL_MM = 0.05                       # notConvergedDistanceThreshold

    pfsDesign = _makeScienceDesign(cc)
    visit     = allocateSimVisitId(db)
    pfsConfig = pfsConfigUtils.pfsConfigFromDesign(pfsDesign, visit, calibModel=cc.calibModel)

    # Targets in cobra order, straight from the config the command would have used.
    targets = pfsConfigUtils.makeTargetsArray(pfsConfig)
    isNan = np.isnan(targets)
    nAll = len(targets)

    rng = np.random.default_rng(7)
    # Bulk: 10 um per axis.  Tail: 6% at 60 um per axis, i.e. reliably past tolerance.
    noise = 0.010 * (rng.standard_normal(nAll) + 1j * rng.standard_normal(nAll))
    tail  = rng.random(nAll) < 0.06
    noise[tail] *= 6.0
    lost  = rng.random(nAll) < 0.02      # MCS never matched a spot

    measured = targets + noise
    measured[isNan] = np.nan             # broken cobras carry no target
    insertSimCobraMatch(db, visit, 0, targets, measured, ~lost)

    maxIter = pfsConfigUtils.finalize(pfsConfig, finalIteration=0,
                                      notConvergedDistanceThreshold=TOL_MM,
                                      atThetas=np.zeros(nAll), atPhis=np.zeros(nAll))
    assert maxIter == 0, f'finalize returned iteration {maxIter}'

    # Re-derive the expectation independently, in cobra order, then compare.
    fiberId    = FiberIds().cobraIdToFiberId(np.arange(1, nAll + 1))
    index      = pd.DataFrame(dict(fiberId=pfsConfig.fiberId,
                                   i=np.arange(len(pfsConfig.fiberId)))).set_index('fiberId')
    idx        = index.loc[fiberId].i.to_numpy()
    status     = pfsConfig.fiberStatus[idx]
    # pfsConfigFromDesign already applies setFiberStatus, so a fresh one gives the
    # pre-convergence status finalize() gates on.
    wasGood    = pfsConfigUtils.pfsConfigFromDesign(
        pfsDesign, visit, calibModel=cc.calibModel).fiberStatus[idx] == FiberStatus.GOOD

    residual   = np.abs(measured - targets)
    inSpec     = residual <= TOL_MM

    # DAMD-195: on the convergence path GOOD is a positive claim about usable science
    # flux, so everything not measurably on target is NOTCONVERGED -- including the
    # cobras MCS lost.  BLACKSPOT belongs to the snapshot path only.
    nGood = int(np.sum(status[wasGood & ~lost & inSpec] == FiberStatus.GOOD))
    nConv = int(np.sum(status[wasGood & ~(inSpec & ~lost)] == FiberStatus.NOTCONVERGED))
    expGood = int((wasGood & ~lost & inSpec).sum())
    expConv = int((wasGood & ~(inSpec & ~lost)).sum())
    print(f'  within spec       -> GOOD          {nGood}/{expGood}')
    print(f'  off target / lost -> NOTCONVERGED  {nConv}/{expConv}')
    print(f'  BLACKSPOT written : {int((status == FiberStatus.BLACKSPOT).sum())} (must be 0)')

    assert expGood and expConv, 'a population is empty; noise model is degenerate'
    assert nGood == expGood, f'{expGood - nGood} in-spec cobras not GOOD'
    assert nConv == expConv, f'{expConv - nConv} off-target cobras not NOTCONVERGED'
    assert not (status == FiberStatus.BLACKSPOT).any(), 'convergence must not write BLACKSPOT'
    assert not (status == FiberStatus.MASKED).any(), 'MASKED is retired'

    # Snapshot path: no threshold, nothing commanded, so detection is the only fact.
    snap = pfsConfigUtils.pfsConfigFromDesign(pfsDesign, visit, calibModel=cc.calibModel)
    pfsConfigUtils.finalize(snap, finalIteration=0,
                            atThetas=np.zeros(nAll), atPhis=np.zeros(nAll))
    snapStatus = snap.fiberStatus[idx]
    nSnapDark = int(np.sum(snapStatus[wasGood & lost] == FiberStatus.BLACKSPOT))
    print(f'  snapshot: undetected -> BLACKSPOT  {nSnapDark}/{int((wasGood & lost).sum())}')
    assert nSnapDark == int((wasGood & lost).sum()), 'snapshot must mark hidden fibres BLACKSPOT'
    assert not (snapStatus == FiberStatus.NOTCONVERGED).any(), 'snapshot cannot say NOTCONVERGED'

    # pfiCenter must carry the measurement, and be NaN exactly where MCS saw nothing.
    assert np.all(np.isnan(pfsConfig.pfiCenter[idx][lost, 0])), 'undetected cobra has a position'
    assert not np.any(np.isnan(pfsConfig.pfiCenter[idx][~lost & ~isNan, 0])), 'detected cobra has NaN'
    print('  pfiCenter NaN exactly where undetected ✓')
    print('  PASS')
    return True


def test_persistedConvergence(cc, physics, db):
    """The full write path: finalize -> FITS -> opdb, with nothing stubbed.

    Every other test replaces _finalizeWriteIngestPfsConfig with a no-op, so the
    validation mask and convergence parameters are computed and then thrown away.  This
    drives the real method and asserts the values survive both a FITS round trip and the
    opdb ingest, which is the only place the datamodel, pfs_utils and opdb schema are
    exercised together.
    """
    print('\n── test_persistedConvergence ──────────────────────────')
    from ics.cobraCharmer import targetValidation
    TOL_MM, REQ_TOL, N_ITER = 0.05, 0.01, 12

    pfsDesign = _makeScienceDesign(cc)
    visit     = allocateSimVisitId(db)
    pfsConfig = pfsConfigUtils.pfsConfigFromDesign(pfsDesign, visit, calibModel=cc.calibModel)
    targets   = pfsConfigUtils.makeTargetsArray(pfsConfig)
    nAll      = len(targets)

    # Same three inputs moveToPfsDesign builds the mask from.
    flags     = targetValidation.validateTargets(cc.calibModel, targets)
    isScience = pfsConfigUtils.getCobraTargetMask(pfsConfig, [TargetType.SCIENCE])
    doNotMove = np.zeros(nAll, dtype=bool)
    doNotMove[::400] = True                 # a handful of operator-masked cobras
    validationMask = pfsConfigUtils.targetValidationMask(flags, isScience)
    command = pfsConfigUtils.cobraCommand(converge=isScience & ~doNotMove,
                                          toBlackDot=~isScience & ~doNotMove)

    fallback = pfsConfigUtils.resolveTargetFallback(
        {'targetFallback': {'invalid': 'BLACKSPOT', 'unassigned': 'BLACKSPOT'}})
    pfsConfig.convergenceParams = pfsConfigUtils.convergenceParams(
        fallback, TOL_MM, False, numIterations=N_ITER, requestedTolerance=REQ_TOL)
    pfsConfig.convergenceParams['elapsedTime'] = 61.25

    rng      = np.random.default_rng(11)
    noise    = 0.010 * (rng.standard_normal(nAll) + 1j * rng.standard_normal(nAll))
    noise[rng.random(nAll) < 0.06] *= 6.0
    detected = rng.random(nAll) >= 0.02
    measured = targets + noise
    measured[np.isnan(targets)] = np.nan
    insertSimCobraMatch(db, visit, 0, targets, measured, detected)

    # NOT_COMMANDED records that fps left the cobra alone, and must hold for every cobra
    # regardless of target type -- otherwise "was it commanded" is unrecoverable from the
    # file for exactly the cobras that were not.  The three actions must also partition.
    notCommanded = command == int(CobraCommand.NOT_COMMANDED)
    assert np.array_equal(notCommanded, doNotMove), (
        f'NOT_COMMANDED on {int(notCommanded.sum())} cobras, {int(doNotMove.sum())} uncommanded')
    counts = {CobraCommand(v).name: int(n) for v, n in zip(*np.unique(command, return_counts=True))}
    assert sum(counts.values()) == nAll, f'commands do not partition the fleet: {counts}'
    assert (validationMask[~isScience] == 0).all(), \
        'a non-science cobra carries a validation bit'
    print(f'  NOT_COMMANDED == ~commanded on all {nAll} cobras ({counts}) \u2713"'.replace('"', ''))

    nRefused = int((validationMask != 0).sum())
    print(f'  refused targets: {nRefused}  '
          f'({dict((f.name, int((validationMask & int(f) != 0).sum())) for f in TargetValidation)})')
    assert nRefused, 'no refused targets; the test would prove nothing'

    fps = makeFakeFpsCmd(cc, db)
    fps.actor.visitor.frameSeq = 1          # finalize reads frameSeq - 1
    fps.atThetas, fps.atPhis = np.zeros(nAll), np.zeros(nAll)
    fps._finalizeWriteIngestPfsConfig(pfsConfig, cmd=FakeCmd(),
                                      notConvergedDistanceThreshold=TOL_MM,
                                      validationMask=validationMask,
                                      cobraCommand=command)

    # ── 1. FITS round trip ────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as dirName:
        pfsConfig.write(dirName=dirName)
        readBack = PfsConfig.read(pfsConfig.pfsDesignId, visit, dirName=dirName)
        np.testing.assert_array_equal(readBack.targetValidationMask,
                                      pfsConfig.targetValidationMask)
        np.testing.assert_array_equal(readBack.cobraCommand, pfsConfig.cobraCommand)
        # the accessor must give back exactly the cobras, in cobra order
        onCobra = readBack.cobraConfig()
        assert np.array_equal(onCobra.cobraId, np.arange(1, pfsConfigUtils.N_COBRAS + 1)), \
            'cobraConfig() is not the cobras in cobraId order'
        np.testing.assert_array_equal(onCobra.cobraCommand, command)
        assert readBack.convergenceParams == pfsConfig.convergenceParams, (
            f'{readBack.convergenceParams} != {pfsConfig.convergenceParams}')
    print(f'  FITS round trip: mask + {len(pfsConfig.convergenceParams)} params intact ✓')

    # ── 2. opdb: the scalars, once per visit ──────────────────────────────────
    row = db.query_dataframe(f"""
        SELECT converg_num_iter, converg_elapsed_time, converg_tolerance,
               converg_distance_threshold, target_fallback_invalid,
               target_fallback_unassigned, fiducial_check_skipped, inst_status_flag
        FROM pfs_config WHERE visit0 = {visit}""")
    assert len(row) == 1, f'{len(row)} pfs_config rows for visit {visit}, expected 1'
    row = row.iloc[0]
    assert int(row.converg_num_iter) == N_ITER, row.converg_num_iter
    assert abs(row.converg_tolerance - REQ_TOL) < 1e-9, row.converg_tolerance
    assert abs(row.converg_distance_threshold - TOL_MM) < 1e-9, row.converg_distance_threshold
    assert abs(row.converg_elapsed_time - 61.25) < 1e-4, row.converg_elapsed_time
    assert row.target_fallback_invalid == 'BLACKSPOT', row.target_fallback_invalid
    assert row.target_fallback_unassigned == 'BLACKSPOT', row.target_fallback_unassigned
    assert bool(row.fiducial_check_skipped) is False, row.fiducial_check_skipped
    print(f'  opdb pfs_config: 1 row, all 8 scalars match ✓')

    # ── 3. opdb: the mask, once per fiber ─────────────────────────────────────
    fibers = db.query_dataframe(f"""
        SELECT fiber_id, target_validation_mask FROM pfs_config_fiber
        WHERE visit0 = {visit} ORDER BY fiber_id""")
    assert len(fibers) == len(pfsConfig.fiberId), (
        f'{len(fibers)} fiber rows, expected {len(pfsConfig.fiberId)}')
    order = np.argsort(pfsConfig.fiberId)
    np.testing.assert_array_equal(fibers.target_validation_mask.to_numpy(),
                                  pfsConfig.targetValidationMask[order])
    inDb = int((fibers.target_validation_mask != 0).sum())
    notSet = int((fibers.target_validation_mask == int(TargetValidation.NOT_SET)).sum())
    print(f'  opdb pfs_config_fiber: {len(fibers)} rows, {inDb} non-zero '
          f'({notSet} NOT_SET, no cobra) ✓')
    print('  PASS')
    return True


def test_genPfsConfigFromMcs(cc, physics, db):
    """genPfsConfigFromMcs — verify exposeAndExtractPositions called."""
    print('\n── test_genPfsConfigFromMcs ───────────────────────────')
    _resetToHome(cc, physics)
    pfsDesign = _makeHomeDesign(cc)
    designId  = pfsDesign.pfsDesignId

    fps = makeFakeFpsCmd(cc, db)
    fps.getPfsConfig = lambda cmd, visit, pfsDesign, maskFile=None: \
        pfsConfigUtils.pfsConfigFromDesign(pfsDesign, visit, calibModel=cc.calibModel)

    buildCalled = []

    class _FakePfsConfig:
        filename = 'fake_pfsConfig.fits'

    def _fake_build(cmd, visit, designId):
        buildCalled.append((visit, designId))
        return _FakePfsConfig()

    fps._buildPfsConfigFromMcs = _fake_build

    fps.genPfsConfigFromMcs(FakeCmd({'designId': FakeKeyword(designId)}))

    assert len(buildCalled) == 1, f'_buildPfsConfigFromMcs called {len(buildCalled)}x (expected 1)'
    print(f'  _buildPfsConfigFromMcs called: visit={buildCalled[0][0]}, '
          f'designId=0x{buildCalled[0][1]:016x} ✓')
    print('  PASS')
    return True


def test_makeMotorMapGroups(cc, physics, db):
    """makeMotorMapGroups phi slow — verify buildPhiMotorMaps called with correct args."""
    print('\n── test_makeMotorMapGroups ────────────────────────────')
    fps           = makeFakeFpsCmd(cc, db)
    fps.loadModel = lambda *a, **kw: None

    buildCalled = []

    def _fake_buildPhi(newXml, steps=None, repeat=None, fast=None, tries=None, homed=None):
        buildCalled.append({'steps': steps, 'repeat': repeat, 'fast': fast})

    _real_buildPhi    = eng.buildPhiMotorMaps
    eng.buildPhiMotorMaps = _fake_buildPhi

    try:
        cmd = FakeCmd({
            'phi':      FakeKeyword(True),
            'slowMap':  FakeKeyword(True),
            'stepsize': FakeKeyword(100),
            'repeat':   FakeKeyword(3),
        })
        fps.makeMotorMapwithGroups(cmd)

        assert len(buildCalled) == 1, f'buildPhiMotorMaps called {len(buildCalled)}x (expected 1)'
        assert buildCalled[0]['steps']  == 100,   f'wrong steps: {buildCalled[0]["steps"]}'
        assert buildCalled[0]['repeat'] == 3,     f'wrong repeat: {buildCalled[0]["repeat"]}'
        assert buildCalled[0]['fast']   == False, f'expected fast=False for slowMap'
        print(f'  buildPhiMotorMaps: steps={buildCalled[0]["steps"]} '
              f'repeat={buildCalled[0]["repeat"]} fast={buildCalled[0]["fast"]} ✓')
    finally:
        eng.buildPhiMotorMaps = _real_buildPhi
        eng.setNormalMode()

    print('  PASS')
    return True


def test_testLoop(cc, physics, db):
    """testIteration — smoke test, verify no exception raised."""
    print('\n── test_testLoop ──────────────────────────────────────')
    _resetToHome(cc, physics)
    fps = makeFakeFpsCmd(cc, db)

    fps.testIteration(FakeCmd({'cnt': FakeKeyword(1)}))
    print('  testIteration(cnt=1) completed without exception ✓')
    print('  PASS')
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Original dot-convergence test (unchanged except SimCobraCoach → real cc)
# ─────────────────────────────────────────────────────────────────────────────

def test_dotConvergence(cc, physics, db):
    """Dot-hiding geometry + pfsConfig from a BLACKSPOT design."""
    print('\n── test_dotConvergence ────────────────────────────────')
    if not HAS_DOT_CONVERGENCE:
        print('  SKIP (dotConvergence not available on this branch)')
        return 'SKIP'

    from ics.fpsActor.utils import dotGeometry

    nAll = len(cc.allCobras)

    # ── 1. Reset to home ─────────────────────────────────────────────────────
    atThetas = (cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (2 * np.pi) + np.pi
    atPhis   = np.zeros(nAll)
    with physics.lock:
        physics.trueThetas[:] = atThetas
        physics.truePhis[:]   = atPhis
    cc.cobraInfo['thetaAngle'] = atThetas.copy()
    cc.cobraInfo['phiAngle']   = atPhis.copy()
    cc.cobraInfo['position']   = cc.pfi.anglesToPositions(cc.allCobras, atThetas, atPhis)

    # ── 2. Dot geometry functions ─────────────────────────────────────────────
    thetaDot, phiCenter, phiMin, phiMax, phiEnter, direction, halfDot = \
        dotGeometry.computeDotAngles(cc)
    nIter = 8
    phiInDot = dotGeometry.computePhiAtFraction(phiCenter, halfDot, direction, 0.1)
    phiStart = dotGeometry.computePhiStart(phiInDot, direction)
    phiRamp  = dotGeometry.computePhiRamp(phiStart, phiEnter, phiInDot, direction, nIter=nIter)

    nCCW = int(np.sum(direction[cc.goodIdx] == 1))
    nCW  = int(np.sum(direction[cc.goodIdx] == -1))
    print(f'  computeDotAngles: {nCCW} CCW, {nCW} CW cobras')

    # Geometry sanity checks
    assert nCCW + nCW == len(cc.goodIdx), 'some goodIdx cobras have direction=0'
    assert np.all(np.isfinite(phiCenter[cc.goodIdx])), 'phiCenter NaN in goodIdx cobras'
    assert np.all(np.isfinite(thetaDot[cc.goodIdx])), 'thetaDot NaN in goodIdx cobras'

    ccwGood = cc.goodIdx[direction[cc.goodIdx] == 1]
    cwGood  = cc.goodIdx[direction[cc.goodIdx] == -1]
    if len(ccwGood):
        assert np.all(phiStart[ccwGood] <= phiEnter[ccwGood] + 1e-9), \
            'CCW phiStart should be <= phiEnter'
    if len(cwGood):
        assert np.all(phiStart[cwGood] >= phiEnter[cwGood] - 1e-9), \
            'CW phiStart should be >= phiEnter'

    phiAt0 = dotGeometry.computePhiAtFraction(phiCenter, halfDot, direction, 0.1)
    phiAt04 = dotGeometry.computePhiAtFraction(phiCenter, halfDot, direction, 0.4)
    assert np.all(np.isfinite(phiAt0[cc.goodIdx])), 'phiAt0 NaN in goodIdx'
    assert np.all(np.isfinite(phiAt04[cc.goodIdx])), 'phiAt04 NaN in goodIdx'
    print('  geometry sanity checks PASS')

    # ── 3. createDotConvergenceDesign ─────────────────────────────────────────
    pfsDesign = pfsDesignUtils.createDotConvergenceDesign(
        cc.calibModel, cc.pfi, cc.allCobras,
        thetaDot, phiStart,
        movingIdx=cc.goodIdx,
        designName='sim_test')
    nBlackspot = int(np.sum(pfsDesign.targetType == TargetType.BLACKSPOT))
    assert nBlackspot == len(cc.goodIdx), \
        f'expected {len(cc.goodIdx)} BLACKSPOT fibers, got {nBlackspot}'
    print(f'  createDotConvergenceDesign OK: {nBlackspot} BLACKSPOT, '
          f'designId=0x{pfsDesign.pfsDesignId:016x}')

    # ── 4. pfsConfig with imperfect convergence ────────────────────────────────
    NOISE_MM   = 0.05
    print('\n  Simulating pfsConfig from imperfect moveToPfsDesign …')
    moveVisitId       = allocateSimVisitId(db)
    nominalPositions  = cc.pfi.anglesToPositions(cc.allCobras, thetaDot, phiStart)
    rng               = np.random.default_rng(42)
    noise             = NOISE_MM * (rng.standard_normal(nAll) + 1j * rng.standard_normal(nAll))
    measuredPositions = nominalPositions + noise

    ctRows = pd.DataFrame({
        'pfs_visit_id':      [moveVisitId] * nAll,
        'iteration':         [0]           * nAll,
        'cobra_id':          list(range(1, nAll + 1)),
        'pfi_nominal_x_mm':  nominalPositions.real.tolist(),
        'pfi_nominal_y_mm':  nominalPositions.imag.tolist(),
        'pfi_target_x_mm':   nominalPositions.real.tolist(),
        'pfi_target_y_mm':   nominalPositions.imag.tolist(),
    })
    db.insert_dataframe('cobra_target', ctRows)

    moveFrameId = moveVisitId * 100
    db.insert_kw('mcs_exposure', mcs_frame_id=moveFrameId, pfs_visit_id=moveVisitId)

    mcsRows = pd.DataFrame({
        'mcs_frame_id':     [moveFrameId] * nAll,
        'spot_id':          list(range(nAll)),
        'mcs_center_x_pix': measuredPositions.real * 100,
        'mcs_center_y_pix': measuredPositions.imag * 100,
    })
    db.insert_dataframe('mcs_data', mcsRows)

    matchRows = pd.DataFrame({
        'pfs_visit_id':    [moveVisitId] * nAll,
        'iteration':       [0]           * nAll,
        'cobra_id':        list(range(1, nAll + 1)),
        'mcs_frame_id':    [moveFrameId] * nAll,
        'spot_id':         list(range(nAll)),
        'pfi_center_x_mm': measuredPositions.real.tolist(),
        'pfi_center_y_mm': measuredPositions.imag.tolist(),
    })
    db.insert_dataframe('cobra_match', matchRows)

    pfsConfig = pfsConfigUtils.pfsConfigFromDesign(pfsDesign, moveVisitId, calibModel=cc.calibModel)
    pfsConfigUtils.finalize(pfsConfig, finalIteration=0)

    goodFiberMask = np.isin(pfsDesign.fiberId,
                            pfsDesign.fiberId[pfsDesign.targetType == TargetType.BLACKSPOT])
    pfiCenter  = pfsConfig.pfiCenter[goodFiberMask]
    pfiNominal = pfsConfig.pfiNominal[goodFiberMask]
    assert not np.all(np.isnan(pfiCenter)), 'pfiCenter all NaN — finalize did not populate it'
    validMask  = ~np.isnan(pfiCenter[:, 0])
    distances  = np.hypot(pfiCenter[validMask, 0] - pfiNominal[validMask, 0],
                          pfiCenter[validMask, 1] - pfiNominal[validMask, 1])
    meanDist   = float(np.nanmean(distances))
    print(f'  pfsConfig OK: mean |pfiCenter-pfiNominal| = {meanDist*1000:.1f} μm '
          f'(expect ~{NOISE_MM*1000:.0f} μm)')
    assert abs(meanDist - NOISE_MM) < NOISE_MM * 0.5, \
        f'mean distance {meanDist:.4f} mm too far from expected {NOISE_MM:.4f} mm'

    print('  PASS')
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    logging.basicConfig(level=logging.WARNING)
    print('=== fps regression simulator (deep end-to-end) ===')

    # Import physics server
    sys.path.insert(0, os.path.dirname(__file__))
    from pfiSimServer import PhysicsState, startPhysicsServer, makeSimExpose

    # 1. Get the correct calibration XML path via pfs.utils.butler.
    #    pfiDesign.loadPfi() uses the older butler.mapPathForModule() which returns
    #    ALL_.xml (with underscore) — the new butler API gives the correct ALL.xml.
    from pfs.utils import butler as _pfsButler
    from ics.cobraCharmer import pfiDesign
    _b = _pfsButler.Butler()
    _xmlPath = pathlib.Path(_b.getPath('moduleXml', moduleName='ALL', version=''))
    print(f'calibration XML: {_xmlPath}')

    # 2. Pre-load calibration model so PhysicsState can build its motor-map LUT
    #    before CobraCoach initiates the TCP connection.
    calibModel = pfiDesign.PFIDesign(_xmlPath)
    calibModel.fixModuleIds()

    # 3. Start physics FPGA server FIRST.
    #    The pfi cobraCharmer has PFI(doConnect=True) by default, so loadModel()
    #    will auto-connect to localhost:4001 during connect().
    physics = PhysicsState(calibModel)
    startPhysicsServer(physics)

    # 4. Create CobraCoach without loading the model (avoids premature connect + wrong path).
    #    Then monkeypatch time.sleep to skip the DIAG/POWER/RESET dance (~3 s of sleeps),
    #    load the model from the correct XML path (which triggers connect() → physics server).
    cc = CobraCoach(loadModel=False)
    _real_sleep = _time.sleep
    _time.sleep = lambda _: None
    cc.loadModel(file=_xmlPath)
    _time.sleep = _real_sleep
    print(f'CobraCoach loaded: {len(cc.allCobras)} cobras, {len(cc.goodIdx)} good')

    # 5. Sync physics to cc's calibModel instance (same data, but same object is cleaner).
    with physics.lock:
        physics.calibModel = cc.calibModel

    # 6. Give cc a stub camera so cc.cam.filePrefix/resetStack checks don't crash.
    cc.cam = FakeCam()

    # 7. DB connection
    db = opdb.OpDB()

    # 8. Patch exposeAndExtractPositions on the instance (the ONLY monkey-patch).
    #    Returns truth positions — no hardware, no DB writes.
    cc.exposeAndExtractPositions = makeSimExpose(physics, cc)

    # 9. Initialise truth state and cc.cobraInfo to home
    _resetToHome(cc, physics)

    # 10. Register cc as engineer module's global CobraCoach
    eng.setCobraCoach(cc)

    tests = [
        ('createDesigns',       test_createDesigns),
        ('cobraMoveAngles',     test_cobraMoveAngles),
        ('cobraMoveSteps',      test_cobraMoveSteps),
        ('moveToHome',          test_moveToHome),
        ('moveToSafePosition',  test_moveToSafePosition),
        ('moveToPfsDesign',     test_moveToPfsDesign),
        ('finalizeFiberStatus', test_finalizeFiberStatus),
        ('persistedConvergence', test_persistedConvergence),
        ('genPfsConfigFromMcs', test_genPfsConfigFromMcs),
        ('makeMotorMapGroups',  test_makeMotorMapGroups),
        ('testLoop',            test_testLoop),
        ('dotConvergence',      test_dotConvergence),
    ]

    results = {}
    for name, fn in tests:
        try:
            ok = fn(cc, physics, db)
            if ok == 'SKIP':
                results[name] = 'SKIP'
            else:
                results[name] = 'PASS' if ok else 'FAIL'
        except Exception as e:
            import traceback
            results[name] = f'ERROR: {e}'
            traceback.print_exc()

    print('\n=== RESULTS ===')
    for name, result in results.items():
        if result == 'PASS':
            icon = '✓'
        elif result == 'SKIP':
            icon = '~'
        else:
            icon = '✗'
        print(f'  {icon} {name}: {result}')

    failed = [v for v in results.values() if v not in ('PASS', 'SKIP')]
    if failed:
        sys.exit(1)
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    run_all()
