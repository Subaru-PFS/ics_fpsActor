"""pfiSimServer — physics-aware FPGA simulator for PFS cobra regression tests.

Replaces the two hardware components that can't run offline:
  1. FPGA board — PhysicsFPGAProtocol extends fpgaSim.FPGAProtocol, decodes RUN command
                  binary, updates truth angles via motor-map inversion.
  2. MCS camera — makeSimExpose() returns a drop-in for cc.exposeAndExtractPositions that
                  reads truth state, inserts opdb rows, and returns complex positions.

Usage in tests:

    from pfiSimServer import PhysicsState, startPhysicsServer, makeSimExpose

    physics = PhysicsState(calibModel)
    startPhysicsServer(physics)

    cc = CobraCoach(fpgaHost='localhost')   # connects over TCP to physics server
    cc.exposeAndExtractPositions = makeSimExpose(physics, db, cc, exposeVisitId)
"""
import asyncio
import datetime
import logging
import struct
import threading
import time

import numpy as np
import pandas as pd

from ics.cobraCharmer import fpgaProtocol as proto
from ics.cobraCharmer.fpgaSim import FPGAProtocol
from ics.fpsActor.utils.alfUtils import sgfm

log = logging.getLogger('pfiSimServer')


# ─────────────────────────────────────────────────────────────────────────────
# Motor-map inversion: steps → angle delta
# ─────────────────────────────────────────────────────────────────────────────

def stepsToAngle(calibModel, cId, stepsMag, isCCW, currentAngle, arm='theta'):
    """Invert motor map: magnitude of steps + direction → angle delta (radians).

    Uses the current angle to look up the cumulative step count from the hard stop,
    then adds/subtracts stepsMag, and inverts back to an angle. This correctly
    handles cobras that are not at a hard stop.

    Args:
        calibModel   : PFIDesign
        cId          : 0-based cobra index
        stepsMag     : unsigned step count (magnitude)
        isCCW        : True = CCW direction (negative angle delta)
        currentAngle : current local angle of this arm (radians), used as baseline
        arm          : 'theta' or 'phi'

    Returns:
        Signed angle delta in radians.
    """
    if stepsMag == 0:
        return 0.0
    if arm == 'theta':
        stepArr = calibModel.negThtSlowSteps[cId] if isCCW else calibModel.posThtSlowSteps[cId]
        angArr  = calibModel.thtOffsets[cId]
    else:
        stepArr = calibModel.negPhiSlowSteps[cId] if isCCW else calibModel.posPhiSlowSteps[cId]
        angArr  = calibModel.phiOffsets[cId]
    # Current cumulative steps from hard stop at currentAngle
    cumFrom = float(np.interp(currentAngle, angArr, stepArr))
    # Target cumulative steps
    cumTarget = cumFrom - stepsMag if isCCW else cumFrom + stepsMag
    # Invert back to angle
    targetAngle = float(np.interp(cumTarget, stepArr, angArr))
    return targetAngle - currentAngle


# ─────────────────────────────────────────────────────────────────────────────
# PhysicsState — thread-safe truth angle store
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsState:
    """Thread-safe truth state shared between the asyncio server and the test thread.

    The FPGA server (asyncio thread) writes to trueThetas/truePhis on every RUN command.
    The expose function (test thread) reads them to compute measured positions.
    """

    def __init__(self, calibModel):
        self.calibModel = calibModel
        nAll = calibModel.nCobras
        # Start at home position (theta CW hard stop, phi CCW hard stop)
        self.trueThetas = ((calibModel.tht1 - calibModel.tht0 + np.pi)
                           % (2 * np.pi) + np.pi).copy()
        self.truePhis   = np.zeros(nAll)
        self.lock       = threading.Lock()
        self._frameCtr  = 0

        # Pre-compute (board, cobra_on_board) → calibModel index for fast decoding.
        # board = (module-1)*2 + 1  (+ 1 if positioner is even-numbered)
        # cobra_on_board = (positionerId - 1) // 2 + 1
        self._cobraLUT = {}
        for cId in range(nAll):
            module     = int(calibModel.moduleIds[cId])
            positioner = int(calibModel.positionerIds[cId])
            board      = (module - 1) * 2 + 1
            if (positioner - 1) % 2 == 1:
                board += 1
            cobra_on_board = (positioner - 1) // 2 + 1
            self._cobraLUT[(board, cobra_on_board)] = cId

    def nextFrameId(self):
        """Return a globally unique, monotonically increasing frame counter."""
        with self.lock:
            self._frameCtr += 1
            return self._frameCtr

    def _thetaMax(self, cId):
        """CW hard-stop angle (unwrapped, radians)."""
        return float(((self.calibModel.tht1[cId] - self.calibModel.tht0[cId] + np.pi)
                      % (2 * np.pi) + np.pi))

    def _phiMax(self, cId):
        """CW hard-stop angle (local, radians)."""
        return float((self.calibModel.phiOut[cId] - self.calibModel.phiIn[cId]) % (2 * np.pi))

    def applyRunCmd(self, data, nCobras):
        """Decode a RUN command payload and update truth angles.

        Angles are kept in unwrapped (local) coordinates:
          theta: 0 = CCW hard stop  →  thetaMax = CW hard stop
          phi:   0 = CCW hard stop  →  phiMax   = CW hard stop

        When a cobra hits a hard stop the angle is simply clamped — same
        behaviour as real hardware (the cobra stalls).

        Each cobra occupies 14 bytes (RunParams.toList layout):
            bytes 0-1  : control word (big-endian uint16)
                         bits  0   : theta direction (1=CCW)
                         bits  1   : phi direction (1=CCW)
                         bits  2   : theta enable
                         bits  3   : phi enable
                         bits  4-10: board id (1..84)
                         bits 11-15: cobra-on-board id (1..29)
            bytes 2-3  : theta pulse width (uint16)
            bytes 4-5  : theta step magnitude (uint16, unsigned)
            bytes 6-7  : theta sleep (uint16)
            bytes 8-9  : phi pulse width (uint16)
            bytes 10-11: phi step magnitude (uint16, unsigned)
            bytes 12-13: phi sleep (uint16)
        """
        with self.lock:
            for i in range(nCobras):
                chunk   = data[i * 14:(i + 1) * 14]
                flags   = struct.unpack('>H', chunk[0:2])[0]
                boardId       = (flags >> 4) & 0x7f
                cobraOnBoard  = (flags >> 11) & 0x1f
                thetaDir_ccw  = bool(flags & 0x1)       # bit 0: CCW=1, CW=0
                phiDir_ccw    = bool((flags >> 1) & 1)   # bit 1
                thetaEn       = bool((flags >> 2) & 1)   # bit 2
                phiEn         = bool((flags >> 3) & 1)   # bit 3
                thetaStepsMag = struct.unpack('>H', chunk[4:6])[0]   # unsigned magnitude
                phiStepsMag   = struct.unpack('>H', chunk[10:12])[0]

                cId = self._cobraLUT.get((boardId, cobraOnBoard))
                if cId is None:
                    log.warning(f'Unknown board={boardId} cobra={cobraOnBoard}, skipping')
                    continue

                if thetaEn and thetaStepsMag:
                    delta = stepsToAngle(self.calibModel, cId, thetaStepsMag,
                                         thetaDir_ccw, self.trueThetas[cId], 'theta')
                    self.trueThetas[cId] = np.clip(
                        self.trueThetas[cId] + delta, 0.0, self._thetaMax(cId))

                if phiEn and phiStepsMag:
                    delta = stepsToAngle(self.calibModel, cId, phiStepsMag,
                                         phiDir_ccw, self.truePhis[cId], 'phi')
                    self.truePhis[cId] = np.clip(
                        self.truePhis[cId] + delta, 0.0, self._phiMax(cId))

    def teleport(self, cIds, thetas, phis):
        """Directly set truth angles for given cobra indices (used for homing)."""
        with self.lock:
            if thetas is not None:
                self.trueThetas[cIds] = thetas
            if phis is not None:
                self.truePhis[cIds] = phis


# ─────────────────────────────────────────────────────────────────────────────
# PhysicsFPGAProtocol — extends fpgaSim to update physics on RUN commands
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsFPGAProtocol(FPGAProtocol):
    """FPGA simulator with cobra physics.

    Overrides processOneCmd() to:
      - On RUN_CMD: decode the payload and apply steps to truth angles via stepsToAngle
      - All other commands: delegate to FPGAProtocol (DIAG, HOUSEKEEPING, SETFREQ, POWER, ADMIN)
    """

    def __init__(self, physics, **kwargs):
        super().__init__(**kwargs)
        self.physics = physics

    def processOneCmd(self):
        """Remove one command from self.data and process it."""
        header, data, self.data = self.splitCommand(self.data)
        cmd = int(header[0])
        self.fpgaLogger.logCommand(header, data)

        if cmd == proto.RUN_CMD:
            # bytes 2-3 in the RUN header encode nCobras (big-endian)
            nCobras = struct.unpack('>H', header[2:4])[0]
            self.physics.applyRunCmd(data, nCobras)

        # All commands respond with the standard ACK
        if cmd == proto.HOUSEKEEPING_CMD:
            self.housekeepingHandler(header, data)
            return
        if cmd == proto.DIAG_CMD:
            self.diagHandler()
            return
        if cmd == proto.POWER_CMD:
            self.powerHandler(header)
            return
        if cmd == proto.ADMIN_CMD:
            self.adminHandler(header, data)
            return
        if cmd == proto.EXIT_CMD:
            self.pleaseFinishLoop = True
        elif cmd == proto.FLUSH_CMD:
            self.resetBuffer()

        self.respond()
        if cmd in {proto.RUN_CMD, proto.CAL_CMD, proto.SETFREQ_CMD}:
            self.respond()


# ─────────────────────────────────────────────────────────────────────────────
# startPhysicsServer — asyncio server in a daemon thread
# ─────────────────────────────────────────────────────────────────────────────

def startPhysicsServer(physics, host='localhost', port=4001):
    """Start PhysicsFPGAProtocol as an asyncio server in a daemon thread.

    Returns the thread (daemon=True, no cleanup needed).
    Blocks for 0.3 s to ensure the server socket is open before returning.
    """
    loop = asyncio.new_event_loop()

    async def _serve():
        server = await loop.create_server(
            lambda: PhysicsFPGAProtocol(physics, debug=False),
            host, port)
        log.info(f'PhysicsFPGAProtocol listening on {host}:{port}')
        await server.serve_forever()

    t = threading.Thread(
        target=lambda: loop.run_until_complete(_serve()),
        daemon=True,
        name='physics-fpga-server')
    t.start()
    time.sleep(0.3)   # let the socket open
    print(f'  PhysicsFPGAProtocol running on {host}:{port}')
    return t


# ─────────────────────────────────────────────────────────────────────────────
# makeSimExpose — replaces cc.exposeAndExtractPositions (the only monkey-patch)
# ─────────────────────────────────────────────────────────────────────────────

DETECTION_RADIUS_MM = 0.711
"""Tip-to-dot-centre distance below which a fibre stops being centroided.

The 50% point of the detection profile measured over 4.44M stacked detections.  The
real transition is not a step -- it has a 54 um width -- but a threshold keeps a run
reproducible, which matters more here than reproducing the last few percent.
"""


def makeSimExpose(physics, cc):
    """Return a replacement for cc.exposeAndExtractPositions.

    Reads truth angles from PhysicsState, computes PFI mm positions, and returns
    them directly. No hardware, no DB writes — the caller (cc.moveSteps) uses the
    returned positions immediately via positionsToAngles.

    A fibre closer than DETECTION_RADIUS_MM to its black dot is reported the way the
    real pipeline reports it: cobraInfo['detected'] goes False, and the position
    returned is the dot centre, which is what cobra_match substitutes when it matches
    no spot.  Callers that trust the position without checking the flag therefore fail
    here the same way they fail on sky, instead of silently working.

    Args:
        physics : PhysicsState (shared with the asyncio server thread)
        cc      : real CobraCoach instance (used for pfi.anglesToPositions)
    """
    dotPos = sgfm.xDot.to_numpy() + 1j * sgfm.yDot.to_numpy()

    def _expose(**kwargs):
        with physics.lock:
            trueT = physics.trueThetas.copy()
            trueP = physics.truePhis.copy()

        positions = np.asarray(cc.pfi.anglesToPositions(cc.allCobras, trueT, trueP)).copy()
        hidden = np.abs(positions - dotPos) < DETECTION_RADIUS_MM
        positions[hidden] = dotPos[hidden]
        cc.cobraInfo['detected'] = ~hidden
        return positions

    return _expose
