import logging
import os
import pathlib
import signal
import subprocess as sub
import sys
import time
from importlib import reload

import cv2
import ics.cobraCharmer.pfiDesign as pfiDesign
import ics.fpsActor.boresightMeasurements as boresightMeasure
import ics.fpsActor.utils.pfsConfig as pfsConfigUtils
import ics.fpsActor.utils.pfsDesign as pfsDesignUtils
import ics.utils.sps.fits as fits
import numpy as np
import opscore.protocols.keys as keys
import opscore.protocols.types as types
import pandas as pd
from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer.cobraCoach import calculation
from ics.cobraCharmer.cobraCoach import cobraCoach
from ics.cobraCharmer.cobraCoach import engineer as eng
from ics.fpsActor import fpsFunction as fpstool
from ics.fpsActor import fpsState
from ics.fpsActor import najaVenator
from ics.fpsActor.utils import display as vis
from ics.fpsActor.utils import dotGeometry
from ics.fpsActor.utils import motorScales
from ics.cobraCharmer import targetValidation
from ics.fpsActor.utils.cobraCenters import updateCobraCenters
from pfs.utils.database import opdb
from pfs.utils import butler
from pfs.utils.pfsConfigUtils import tweakTargetPosition
from pfs.utils.versions import collectVersions
import ics.utils.cmd as cmdUtils
from pfs.datamodel import TargetType

reload(vis)

reload(calculation)
reload(pfiControl)
reload(cobraCoach)
reload(najaVenator)
reload(eng)

reload(pfsConfigUtils)



class FpsCmd(object):
    def __init__(self, actor):
        # This lets us access the rest of the actor.
        self.actor = actor

        self.nv = najaVenator.NajaVenator()

        self.tranMatrix = None
        self.dotScan = None  # in-memory dot-scan step handed off from moveToPfsDesign to moveToDotByFlux
        # Declare the commands we implement. When the actor is started
        # these are registered with the parser, which will call the
        # associated methods when matched. The callbacks will be
        # passed a single argument, the parsed and typed command.
        #
        self.vocab = [
            ('ping', '', self.ping),
            ('status', '', self.status),
            ('reset', '[<mask>]', self.reset),
            ('power', '[<mask>]', self.power),
            ('powerOn', '', self.powerOn),
            ('powerOff', '', self.powerOff),
            ('diag', '', self.diag),
            ('calib', '[<board>] [@updateModel] [@ccw]', self.calib),
            ('hk', '[<board>] [@short]', self.hk),
            ('writeModel', '<xml>', self.writeModel),
            ('connect', '', self.connect),
            ('fpgaSim', '@(on|off) [<datapath>]', self.fpgaSim),
            ('ledlight', '@(on|off)', self.ledlight),
            ('loadDesign', '<id>', self.loadDesign),
            ('loadModel', '[<xml>]', self.loadModel),
            ('cobraAndDotRecenter', '', self.cobraAndDotRecenter),
            ('movePhiForThetaOps', '<runDir>', self.movePhiForThetaOps),
            ('movePhiForDots', '<angle> <iteration> [<visit>]', self.movePhiForDots),
            ('movePhiToAngle', '<angle> <iteration> [<visit>]', self.movePhiToAngle),

            ('createHomeDesign', '[@(phi|theta|all)] [<maskFile>] [<designName>]', self.createHomeDesign),
            ('createBlackDotDesign', '[<maskFile>] [<designName>]', self.createBlackDotDesign),
            ('createDotConvergenceDesign', '[<maskFile>] [<designName>]', self.createDotConvergenceDesign),
            ('createThetaPhiScanDesign', '<thetaAngle> <phiAngle> [<designName>]', self.createThetaPhiScanDesign),
            ('genPfsConfigFromMcs', '<visit> <designId> [<expTime>]', self.genPfsConfigFromMcs),
            ('moveToHome', '@(phi|theta|all) [<expTime>] [@noMCSexposure] [<visit>] [<maskFile>] '
                           '[<designId>] [@thetaCCW]', self.moveToHome),

            ('setCobraMode', '@(phi|theta|normal)', self.setCobraMode),
            ('setGeometry', '@(phi|theta) <runDir>', self.setGeometry),
            ('moveToPfsDesign',
             '<designId> [@twoStepsOff] [@shortExpOff] [@goHome] [@noTweak] [@skipFiducialInterferenceCheck] [@allowMaskedCobras] [<visit>] [<expTime>] [<iteration>] [<tolerance>] [<maskFile>]',
             self.moveToPfsDesign),
            ('moveToSafePosition', '[<expTime>] [<visit>] [<tolerance>] [<phiAngle>] [<thetaAngle>] [@noHome]', self.moveToSafePosition),
            ('makeMotorMap', '@(phi|theta) <stepsize> <repeat> [<totalsteps>] [@slowOnly] [@forceMove] [<visit>]',
             self.makeMotorMap),
            ('makeMotorMapGroups', '@(phi|theta) <stepsize> <repeat> [@(slowMap|fastMap)] [<cobraGroup>] [<visit>]',
             self.makeMotorMapwithGroups),
            ('makeOntimeMap', '@(phi|theta) [<visit>]', self.makeOntimeMap),
            ('angleConverge', '@(phi|theta) <angleTargets> [<visit>]', self.angleConverge),
            ('targetConverge', '@(ontime|speed) <totalTargets> <maxsteps> [<visit>]', self.targetConverge),
            ('motorOntimeSearch', '@(phi|theta) [<visit>]', self.motorOntimeSearch),
            ('calculateBoresight', '[<startFrame>] [<endFrame>] [@writeToDB]', self.calculateBoresight),
            ('testCamera', '[<visit>]', self.testCamera),
            ('testIteration', '[<visit>] [<expTime>] [<cnt>]', self.testIteration),
            ('expose', '[<visit>] [<expTime>] [<cnt>]', self.testIteration),  # New alias
            ('testLoop', '[<visit>] [<expTime>] [<cnt>] [@noMatching]',
             self.testIteration),  # Historical alias.
            ('cobraMoveSteps', '@(phi|theta) <stepsize> [<maskFile>] [<applyScaling>] [<cnt>]', self.cobraMoveStepsCmd),
            ('cobraMoveAngles','@(phi|theta) <angle> [<maskFile>] [<visit>] [<designId>] [<expTime>] [@(genPfsConfig)]',
             self.cobraMoveAngles),
            ('loadDotScales', '[<filename>]', self.loadDotScales),
            ('updateDotLoop', '<filename> [<stepsPerMove>] [@noMove]', self.updateDotLoop),
            ('testDotMove', '[<stepsPerMove>]', self.testDotMove),
            ('setDb', '[<host>] [<user>] [<port>] [<dbname>]', self.setDb),
            ('updateCobrasCenters', '[@brokenOnly]', self.updateCobrasCenters),
            ('moveToDotByFlux', '[<nRemaining>]', self.moveToDotByFlux)
        ]

        # Define typed command arguments for the above commands.
        self.keys = keys.KeysDictionary("fps_fps", (1, 1),
                                        keys.Key("cnt", types.Int(), help="times to run loop"),
                                        keys.Key("angle", types.Int(), help="arm angle"),
                                        keys.Key("designId", types.Long(), help="PFS design ID"),
                                        keys.Key("stepsize", types.Int(), help="step size of motor"),
                                        keys.Key("totalsteps", types.Int(), help="total step for motor"),
                                        keys.Key("cobraGroup", types.Int(),
                                                 help="cobra group for avoid collision"),
                                        keys.Key("repeat", types.Int(),
                                                 help="number of iteration for motor map generation"),
                                        keys.Key("angleTargets", types.Int(),
                                                 help="Target number for angle convergence"),
                                        keys.Key("totalTargets", types.Int(),
                                                 help="Target number for 2D convergence"),
                                        keys.Key("maxsteps", types.Int(),
                                                 help="Maximum step number for 2D convergence test"),
                                        keys.Key("xml", types.String(), help="XML filename"),
                                        keys.Key("datapath", types.String(),
                                                 help="Mock data for simulation mode"),
                                        keys.Key("runDir", types.String(), help="Directory of run data"),
                                        keys.Key("startFrame", types.Int(),
                                                 help="starting frame for boresight calculating"),
                                        keys.Key("endFrame", types.Int(),
                                                 help="ending frame for boresight calculating"),
                                        keys.Key("visit", types.Int(), help="PFS visit to use"),
                                        keys.Key("frameId", types.Int(), help="PFS Frame ID"),
                                        keys.Key("iteration", types.Int(), help="Interation number"),
                                        keys.Key("overshootFraction", types.Float(),
                                                 help="fraction past dot center toward outer edge (0=center, 1=outer edge)"),
                                        keys.Key("tolerance", types.Float(), help="Tolerance distance in mm"),
                                        keys.Key("id", types.Long(),
                                                 help="pfsDesignId, to define the target fiber positions"),
                                        keys.Key("maskFile", types.String(), help="mask filename for cobra"),
                                        keys.Key("mask", types.Int(), help="mask for power and/or reset"),
                                        keys.Key("expTime", types.Float(), help="Seconds for exposure"),
                                        keys.Key("theta", types.Float(), help="Distance to move theta"),
                                        keys.Key("phi", types.Float(), help="Distance to move phi"),
                                        keys.Key("thetaAngle", types.Float() ,units='deg',
                                                 help="Angle (deg) to move theta to"),
                                        keys.Key("phiAngle", types.Float(), units='deg',
                                                 help="Angle (deg) to move phi to"),
                                        keys.Key("board", types.Int(), help="board index 1-84"),
                                        keys.Key("stepsPerMove", types.Int(),
                                                 help="number of steps per move"),
                                        keys.Key("applyScaling", types.String(),
                                                 help="scaling filename for cobra"),

                                        keys.Key("host", types.String(), help="opdb hostname"),
                                        keys.Key("user", types.String(), help="opdb user name"),
                                        keys.Key("dbname", types.String(), help="opdb db name"),
                                        keys.Key("port", types.Int(), help="opdb port"),
                                        keys.Key("designName", types.String(), help="pfsDesign name"),
                                        keys.Key("dotTarget", types.String(), help="dotTarget"),
                                        keys.Key("nRemaining", types.Int()),
                                        )

        self.logger = logging.getLogger('fps')
        self.logger.setLevel(logging.INFO)

        self.fpgaHost = 'fpga'
        self.p = None
        self.simDataPath = None
        self._db = None

        self.xml = None

        if self.cc is not None:
            eng.setCobraCoach(self.cc)

        self.atPhis = None
        self.atThetas = None

    # .cc and .db live in the actor, so that we can reload safely.
    @property
    def cc(self):
        return self.actor.cc

    @cc.setter
    def cc(self, newValue):
        self.actor.cc = newValue

    def setDb(self, cmd):
        """Set parts of the db URI.

        Override the pfs_instadata config parts of the db URI. If not are set
        reverts to the default values.
        """
        config = self.actor.actorConfig
        cmdKeys = cmd.cmd.keywords

        if 'user' in cmdKeys:
            user = str(cmdKeys['user'].values[0])
        else:
            user = config['db']['user']

        if 'host' in cmdKeys:
            host = str(cmdKeys['host'].values[0])
        else:
            host = config['db']['host']

        if 'port' in cmdKeys:
            port = int(cmdKeys['port'].values[0])
        else:
            port = config['db']['port']

        if 'dbname' in cmdKeys:
            dbname = str(cmdKeys['dbname'].values[0])
        else:
            dbname = config['db']['dbname']

        opdb.OpDB.set_default_connection(host=host,
                                         user=user,
                                         port=port,
                                         dbname=dbname)
        dbConfig = (user, host, port, dbname)

        cmd.finish(f'text="set db config to {dbConfig}')

    def connectToDB(self, cmd=None):
        """connect to the database if not already connected.

        ALL code should use this method to connect to the database."""

        if self._db is not None:
            return self._db

        if cmd is None:
            cmd = self.actor.bcast

        try:
            self._db = opdb.OpDB()
        except Exception as e:
            raise RuntimeError(f"unable to connect to the database: {e}")

        if cmd is not None:
            cmd.inform(f'text="Connected to Database at {self._db.dsn}"')

        return self._db

    def fpgaSim(self, cmd):
        """Turn on/off simulalation mode of FPGA"""
        cmdKeys = cmd.cmd.keywords
        datapath = cmd.cmd.keywords['datapath'].values[0] if 'datapath' in cmdKeys else None

        simOn = 'on' in cmdKeys
        simOff = 'off' in cmdKeys

        my_env = os.environ.copy()

        if simOn is True:
            self.fpgaHost = 'localhost'
            self.logger.info(f'Starting a FPGA simulator.')
            self.p = sub.Popen(['fpgaSim'], env=my_env)

            self.logger.info(f'FPGA simulator started with PID = {self.p.pid}.')
            if datapath is None:
                self.logger.warn(f'FPGA simulator is ON but datapath is not given.')
            self.simDataPath = datapath

        if simOff is True:
            self.fpgaHost = 'fpga'

            self.logger.info(f'Stopping FPGA simulator.')
            self.simDataPath = None

            os.kill(self.p.pid, signal.SIGKILL)
            os.kill(self.p.pid + 1, signal.SIGKILL)

        cmd.finish(f"text='fpgaSim command finished.'")

    def loadModel(self, cmd, doFinish=True):
        """ Loading cobra Model"""
        cmdKeys = cmd.cmd.keywords
        xml = cmdKeys['xml'].values[0] if 'xml' in cmdKeys else None

        # Decide which XML to use
        if xml is not None:
            # If there is a new XML specified, use it
            cmd.inform(f'text="Using specified XML file: {xml}"')
            self.xml = pathlib.Path(xml)
        elif self.xml is not None:
            # If no new XML is specified but a previously loaded XML exists, use the loaded one
            xml = str(self.xml)  # Convert to string for later use
            cmd.inform(f'text="Using previously loaded XML file: {xml}"')
        else:
            # If both are None, load default XML
            butlerResource = butler.Butler()
            xml = butlerResource.getPath("moduleXml", moduleName="ALL", version="")
            self.xml = pathlib.Path(xml)
            cmd.inform(f'text="Loading default XML file: {xml}"')

        self.logger.info(f'Input XML file = {xml}')

        cmd.inform(f"text='Connecting to %s FPGA, simDataPath=%s'" % ('real' if self.fpgaHost == 'fpga' else 'simulator',
                                                                      self.simDataPath))
        if self.simDataPath is None:
            self.cc = cobraCoach.CobraCoach(self.fpgaHost, loadModel=False, actor=self.actor, cmd=cmd)
        else:
            self.cc = cobraCoach.CobraCoach(self.fpgaHost, loadModel=False, simDataPath=self.simDataPath,
                                            actor=self.actor, cmd=cmd)

        self.cc.loadModel(file=pathlib.Path(self.xml))
        eng.setCobraCoach(self.cc)

        diag = cmd.finish if doFinish else cmd.inform
        diag(f"text='Loaded model = {self.xml}'")

    def writeModel(self, cmd):
        """Save current model to XML file"""

        cmdKeys = cmd.cmd.keywords
        xml = cmdKeys['xml'].values[0]

        cmd.inform('text="writing {xml}..."')
        self.cc.calibModel.createCalibrationFile(xml)

        cmd.finish()

    def getPositionsForFrame(self, frameId):
        mcsData = self.nv.readCentroid(frameId)
        self.logger.info(f'mcs data {mcsData.shape[0]}')
        centroids = {'x': mcsData['centroidx'].values.astype('float'),
                     'y': mcsData['centroidy'].values.astype('float')}
        return centroids

    @staticmethod
    def dPhiAngle(target, source, doWrap=False, doAbs=False):
        d = np.atleast_1d(target - source)

        if doAbs:
            d[d < 0] += 2 * np.pi
            d[d >= 2 * np.pi] -= 2 * np.pi

            return d

        if doWrap:
            lim = np.pi
        else:
            lim = 2 * np.pi

        # d[d > lim] -= 2*np.pi
        d[d < -lim] += 2 * np.pi

        return d

    @staticmethod
    def _fullAngle(toPos, fromPos=None):
        """ Return ang of vector, 0..2pi """
        if fromPos is None:
            fromPos = 0 + 0j
        a = np.angle(toPos - fromPos)
        if np.isscalar(a):
            if a < 0:
                a += 2 * np.pi
            if a >= 2 * np.pi:
                a -= 2 * np.pi
        else:
            a[a < 0] += 2 * np.pi
            a[a >= 2 * np.pi] -= 2 * np.pi

        return a

    def ping(self, cmd):
        """Query the actor for liveness/happiness."""

        cmd.finish("text='Present and (probably) well'")

    def status(self, cmd):
        """Report status and version; obtain and send current data"""

        self.actor.sendVersionKey(cmd)

        keyStrings = ['text="FPS Actor status report"']
        keyMsg = '; '.join(keyStrings)

        cmd.inform(keyMsg)
        cmd.diag(sys.path)
        cmd.diag('text="FPS ready to go."')
        cmd.finish()

    def _loadPfsDesign(self, cmd, designId):
        """ Return the pfsDesign for the given pfsDesignId. """

        cmd.warn(f'text="have a pfsDesignId={designId:#016x}, but do not know how to fetch it yet."')

        return None

    def reset(self, cmd):
        """Send the FPGA POWer command with a reset mask. """

        cmdKeys = cmd.cmd.keywords
        resetMask = cmdKeys['mask'].values[0] if 'mask' in cmdKeys else 0x3f

        self.cc.pfi.reset(resetMask)
        time.sleep(1)
        res = self.cc.pfi.diag()
        cmd.inform(f'text="diag = {res}"')

        self.loadModel(cmd)
        cmd.inform(f'text="Reload the XML file and connect to FPGA"')

        cmd.finish(f'text="XML = {self.xml}"')

    def power(self, cmd):
        """Send the FPGA POWer command with a sector mask. """

        cmdKeys = cmd.cmd.keywords
        powerMask = cmdKeys['mask'].values[0] if 'mask' in cmdKeys else 0x0

        self.cc.pfi.power(powerMask)
        time.sleep(1)
        res = self.cc.pfi.diag()
        cmd.finish(f'text="diag = {res}"')

    def hk(self, cmd):
        """Fetch FPGA HouseKeeing info for a board or entire PFI. """

        cmdKeys = cmd.cmd.keywords
        boards = [cmdKeys['board'].values[0]] if 'board' in cmdKeys else range(1, 85)
        short = 'short' in cmdKeys

        for b in boards:
            ret = self.cc.pfi.boardHk(b)
            error, t1, t2, v, f1, c1, f2, c2 = ret
            cmd.inform(f'text="board {b} error={error} temps=({t1:0.2f}, {t2:0.2f}) voltage={v:0.3f}"')
            if not short:
                for cobraId in range(len(f1)):
                    cmd.inform(f'text="    {cobraId + 1:2d}  {f1[cobraId]:0.2f} {c1[cobraId]:0.2f}    '
                               f'{f2[cobraId]:0.2f} {c2[cobraId]:0.2f}"')
        cmd.finish()

    def calib(self, cmd):
        """Run FPGA Piezo tuning info for one board or the entire PFI. """

        cmdKeys = cmd.cmd.keywords
        boards = [cmdKeys['board'].values[0]] if 'board' in cmdKeys else range(1, 85)
        clockwise = 'ccw' not in cmdKeys
        updateModel = 'updateModel' in cmdKeys

        # First, run the calibration step per-board, just to keep current requirements down.
        for b in boards:
            cmd.inform(f'text="calibrating board {b} {"cw" if clockwise else "ccw"} piezo frequencies"')
            ret = self.cc.pfi.calibrateFreq(board=b,
                                            clockwise=clockwise)
            time.sleep(1)

        # Then grab all the results at once.
        for b in boards:
            ret = self.cc.pfi.boardHk(b, updateModel=updateModel)
            error, t1, t2, v, f1, c1, f2, c2 = ret
            cmd.inform(f'text="board {b} error={error} temps=({t1:0.2f}, {t2:0.2f}) voltage={v:0.3f}"')
            for cobraId in range(len(f1)):
                cmd.inform(f'text="    {cobraId + 1:2d}  {f1[cobraId]:0.2f} {c1[cobraId]:0.2f}    '
                           f'{f2[cobraId]:0.2f} {c2[cobraId]:0.2f}"')
        cmd.finish()

    def powerOn(self, cmd):
        """Do what is required to power on all PFI sectors. """

        cmdKeys = cmd.cmd.keywords

        self.cc.pfi.power(0x0)
        time.sleep(1)
        self.cc.pfi.reset()
        time.sleep(1)
        res = self.cc.pfi.diag()
        cmd.inform(f'text="diag = {res}"')

        self.loadModel(cmd)

        cmd.inform(f'text="Reload the XML file and connect to FPGA"')

        cmd.finish(f'text="XML = {self.xml}"')

    def powerOff(self, cmd):
        """Do what is required to power off all PFI sectors """

        cmdKeys = cmd.cmd.keywords

        self.cc.pfi.power(0x23f)
        time.sleep(10)
        res = self.cc.pfi.diag()
        cmd.finish(f'text="diag = {res}"')

    def diag(self, cmd):
        """Read the FPGA sector inventory"""

        cmdKeys = cmd.cmd.keywords

        res = self.cc.pfi.diag()
        cmd.finish(f'text="diag = {res}"')

    def disconnect(self, cmd):
        pass

    def connect(self, cmd):
        """Connect to the FPGA and set up output tree. """

        cmdKeys = cmd.cmd.keywords

        #Power off the FPGA
        self.cc.pfi.power(0x23f)
        time.sleep(2)
        res = self.cc.pfi.diag()
        cmd.inform(f'text="diag = {res}"')

        # Poweer on the FPGA 
        self.cc.pfi.power(0x0)
        time.sleep(1)
        self.cc.pfi.reset()
        time.sleep(1)
        res = self.cc.pfi.diag()
        cmd.inform(f'text="diag = {res}"')

        if self.xml is None:
            butlerResource = butler.Butler()
            xml = butlerResource.getPath("moduleXml", moduleName="ALL", version="")
            self.logger.info(f'Input XML file = {xml}')
            self.xml = pathlib.Path(xml)
            self.logger.info(f'Loading default XML file: {self.xml}')
            cmd.inform(f'text="Using default XML file: {self.xml}"')
        else:
            cmd.inform(f'text="Using previously loaded XML file: {self.xml}"')

        self.logger.info(f'Input XML file = {self.xml}')
        cmd.inform(f"text='Connecting to %s FPGA'" % ('real' if self.fpgaHost == 'fpga' else 'simulator'))
        self.cc = cobraCoach.CobraCoach(self.fpgaHost, loadModel=False, actor=self.actor, cmd=cmd)
        self.cc.loadModel(file=self.xml)
        eng.setCobraCoach(self.cc)

        cmd.finish(f"text='FPGA connected with model = {self.xml}'")


    def ledlight(self, cmd):
        """Turn on/off the fiducial fiber light"""
        cmdKeys = cmd.cmd.keywords

        light_on = 'on' in cmdKeys
        light_off = 'off' in cmdKeys

        if light_on:
            cmdString = f'led on'
            infoString = 'Turn on fiducial fibers'

        else:
            cmdString = f'led off'
            infoString = 'Turn off fiducial fibers'

        cmdVar = self.actor.cmdr.call(actor='peb', cmdStr=cmdString,
                                      forUserCmd=cmd, timout=10)

        self.logger.info(f'{infoString}')

    def loadDesign(self, cmd):
        """ Load our design from the given pfsDesignId. """

        designId = cmd.cmd.keywords['id'].values[0]

        try:
            design = self._loadPfsDesign(cmd, designId)
        except Exception as e:
            cmd.fail(f'text="Failed to load pfsDesign for pfsDesignId={designId}: {e}"')
            return

        fpsState.fpsState.setDesign(designId, design)
        cmd.finish(f'pfsDesignId={designId:#016x}')

    def getCobras(self, cobs):
        # cobs is 0-indexed list
        if cobs is None:
            cobs = np.arange(len(self.allCobras))

        # assumes module == 1 XXX
        return np.array(pfiControl.PFI.allocateCobraList(zip(np.full(len(cobs), 1), np.array(cobs) + 1)))

    def setCobraMode(self, cmd):
        cmdKeys = cmd.cmd.keywords

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys
        normal = 'normal' in cmdKeys

        if phi is True:
            eng.setPhiMode()
            self.logger.info(f'text="Cobra is now in PHI mode"')

        if theta is True:
            eng.setThetaMode()
            self.logger.info(f'text="Cobra is now in THETA mode"')

        if normal is True:
            eng.setNormalMode()
            self.logger.info(f'text="Cobra is now in NORMAL mode"')

        cmd.finish(f"text='Setting cobra mode is finished'")

    def setGeometry(self, cmd):

        cmdKeys = cmd.cmd.keywords
        runDir = pathlib.Path(cmd.cmd.keywords['runDir'].values[0])

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys
        if phi is True:
            eng.setPhiMode()
            self.cc.setPhiGeometryFromRun(runDir)
            self.logger.info(f'Using PHI geometry from {runDir}')
        else:
            eng.setThetaMode()

            center = np.load('/data/MCS/20210918_013/data/theta_center.npy')
            ccwHome = np.load('/data/MCS/20210918_013/data/ccwHome.npy')
            cwHome = np.load('/data/MCS/20210918_013/data/cwHome.npy')

            self.cc.setThetaGeometry(center, ccwHome, cwHome, angle=0)
            self.logger.info(f'Using THETA geometry from preset data')

            # self.cc.setThetaGeometryFromRun(runDir)
            # self.logger.info(f'Using THETA geometry from {runDir}')
        cmd.finish(f"text='Setting geometry is finished'")

    def testCamera(self, cmd):
        """Test camera and non-motion data: we do not provide target data or request match table """

        visit = self.actor.visitor.setOrGetVisit(cmd)
        frameNum = self.actor.visitor.getNextFrameNum()
        cmd.inform(f'text="frame={frameNum}"')
        ret = self.actor.cmdr.call(actor='mcs',
                                   cmdStr=f'expose object expTime=1.0 frameId={frameNum} noCentroid',
                                   forUserCmd=cmd, timeLim=30)
        if ret.didFail:
            raise RuntimeError("mcs expose failed")

        cmd.finish(f'text="camera ping={ret}"')

    def testIteration(self, cmd, doFinish=True):
        """Test camera and all non-motion data: we provide target table data """

        cmdKeys = cmd.cmd.keywords
        visit = self.actor.visitor.setOrGetVisit(cmd)
        cnt = cmdKeys["cnt"].values[0] \
            if 'cnt' in cmdKeys \
            else 1

        expTime = cmdKeys["expTime"].values[0] \
            if "expTime" in cmdKeys \
            else None

        if expTime is not None:
            self.cc.expTime = expTime

        for i in range(cnt):
            frameSeq = self.actor.visitor.frameSeq
            cmd.inform(f'text="taking frame {visit}.{frameSeq} ({i + 1}/{cnt}) and measuring centroids."')
            try:
                pos = self.cc.exposeAndExtractPositions(exptime=expTime)
            except RuntimeError:
                if not cmd.isAlive(): # failure already reported in cobraCoach.
                    return
                raise

            cmd.inform(f'text="found {len(pos)} spots in {visit}.{frameSeq} "')

        if doFinish:
            cmd.finish()

    def cobraMoveAngles(self, cmd):
        """Move cobra in angle. """
        visit = self.actor.visitor.setOrGetVisit(cmd)

        cmdKeys = cmd.cmd.keywords

        # Switch from default no centroids to default do centroids
        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys
        maskFile = cmdKeys['maskFile'].values[0] if 'maskFile' in cmdKeys else None
        designId = cmdKeys['designId'].values[0] if 'designId' in cmdKeys else None
        expTime = cmdKeys['expTime'].values[0] if 'expTime' in cmdKeys else None
        doGenPfsConfig = 'genPfsConfig' in cmdKeys
        # loading mask file and moving only cobra with bitMask==1
        goodIdx = self.loadGoodIdx(maskFile)

        cobras = self.cc.allCobras[goodIdx]

        angles = cmdKeys['angle'].values[0]

        if phi:
            phiMoveAngle = np.deg2rad(np.full(2394, angles))[goodIdx]
            thetaMoveAngle = None
        else:
            phiMoveAngle = None
            thetaMoveAngle = np.deg2rad(np.full(2394, angles))[goodIdx]

        self.cc.moveDeltaAngles(cobras, thetaMoveAngle,
                                phiMoveAngle, thetaFast=False, phiFast=False)

        self.atThetas = self.cc.cobraInfo['thetaAngle'].copy()
        self.atPhis = self.cc.cobraInfo['phiAngle'].copy()

        if doGenPfsConfig:
            self._buildPfsConfigFromMcs(cmd, visit=visit, designId=designId)

        cmd.finish('text="cobraMoveAngles completed"')

    def cobraMoveStepsCmd(self, cmd):
        """Move single cobra in steps. """
        cmdKeys = cmd.cmd.keywords

        # Switch from default no centroids to default do centroids
        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys
        maskFile = cmdKeys['maskFile'].values[0] if 'maskFile' in cmdKeys else None
        applyScaling = cmdKeys['applyScaling'].values[0] if 'applyScaling' in cmdKeys else False
        stepsize = cmd.cmd.keywords['stepsize'].values[0]
        cnt = cmdKeys['cnt'].values[0] if 'cnt' in cmdKeys else 1

        self.cobraMoveSteps(maskFile=maskFile, applyScaling=applyScaling, stepsize=stepsize, theta=theta, phi=phi,
                            nreps=cnt)

        cmd.finish(f'text="cobraMoveSteps stepsize = {stepsize} completed"')

    def cobraMoveSteps(self, maskFile, stepsize, applyScaling=None, theta=False, phi=False, nreps=1):
        theta = False if phi else theta
        # loading mask file and moving only cobra with bitMask==1
        goodIdx = self.loadGoodIdx(maskFile)

        if applyScaling:
            scalingDf = pd.read_csv(applyScaling, index_col=0).sort_values('cobraId')
            if 'steps' in scalingDf.columns:
                stepsize = 1
                scaling = scalingDf.steps.to_numpy()
            else:
                column = 'scaling1' if stepsize > 0 else 'scaling2'
                scaling = scalingDf[column].to_numpy()
        else:
            scaling = np.ones(len(self.cc.allCobras))

        # cobraList = np.array([1240,2051,2262,2278,2380,2393])-1
        cobras = self.cc.allCobras[goodIdx]
        scaling = scaling[goodIdx]

        thetaSteps = np.ones(len(cobras))
        phiSteps = np.ones(len(cobras))

        if theta:
            self.logger.info(f'theta arm is activated, moving {stepsize} steps, applyScaling={applyScaling}')
            thetaSteps *= (scaling*stepsize)
            thetaSteps = thetaSteps.round().astype('int32')
        else:
            self.logger.info(f'phi arm is activated, moving {stepsize} steps, applyScaling={applyScaling}')
            phiSteps *= (scaling*stepsize)
            phiSteps = phiSteps.round().astype('int32')

        t0 = time.time()
        for i in range(nreps):
            t1 = time.time()
            self.cc.pfi.moveSteps(cobras, thetaSteps, phiSteps, thetaFast=False, phiFast=False)
            t2 = time.time()
            self.logger.info(f'moveSteps {i+1}/{nreps}: {t2-t1:0.3f} {t2-t0:0.3f}')

    def makeMotorMapwithGroups(self, cmd):
        """
            Making theta and phi motor map in three groups for avoiding dots.
        """
        cmdKeys = cmd.cmd.keywords

        repeat = cmd.cmd.keywords['repeat'].values[0]
        stepsize = cmd.cmd.keywords['stepsize'].values[0]
        visit = self.actor.visitor.setOrGetVisit(cmd)

        # Setting MCS 'fMethod' to 'previous'
        cmdString = 'switchFMethod fMethod=previous'
        cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                      forUserCmd=cmd, timeLim=60)
        if cmdVar.didFail:
            cmd.fail(f'text="Setting MCS fMethod failed: {cmdUtils.interpretFailure(cmdVar)}"')
            return

        slowMap = 'slowMap' in cmdKeys
        fastMap = 'fastMap' in cmdKeys

        if not (slowMap or fastMap):
            raise RuntimeError('slowMap or fastMap argument needs to be provided')

        # Switch from default no centroids to default do centroids
        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        day = time.strftime('%Y-%m-%d')
        if phi:
            cmd.inform(f'text="Build phi motor map AT ONCE for avoiding dots"')
            eng.setPhiMode()

            if slowMap:
                newXml = f'{day}-phi-slow.xml'
                cmd.inform(f'text="Slow motor map is {newXml}"')

            elif fastMap:
                newXml = f'{day}-phi-fast.xml'
                cmd.inform(f'text="Fast motor map is {newXml}"')

            eng.buildPhiMotorMaps(newXml, steps=stepsize, repeat=repeat, fast=fastMap,
                                      tries=12, homed=False)

        elif theta:
            group = cmd.cmd.keywords['cobraGroup'].values[0]
            cmd.inform(f'text="Build theta motor map in groups for avoiding dots"')

            if slowMap:
                newXml = f'{day}-theta-slow.xml'
                cmd.inform(f'text="Slow motor map is {newXml}"')
                
            elif fastMap:
                newXml = f'{day}-theta-fast.xml'
                cmd.inform(f'text="Fast motor map is {newXml}"')

            eng.buildThetaMotorMaps(newXml, steps=stepsize, group=group, repeat=repeat,
                                        fast=fastMap, tries=12, homed=False)


        # Switching MCS 'fMethod' back to 'previous'
        cmdString = 'switchFMethod fMethod=target'
        cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                      forUserCmd=cmd, timeLim=60)
        if cmdVar.didFail:
            cmd.fail(f'text="Setting MCS fMethod failed: {cmdUtils.interpretFailure(cmdVar)}"')
            raise RuntimeError(f'FAILED to setting mcs FiberID mode!')

        # reloading default xml file
        self.loadModel(cmd, doFinish=False)
        cmd.finish(f'text="Motor map sequence finished"')

    def makeMotorMap(self, cmd):
        """ Making motor map. """
        cmdKeys = cmd.cmd.keywords

        # self._connect()
        repeat = cmd.cmd.keywords['repeat'].values[0]
        stepsize = cmd.cmd.keywords['stepsize'].values[0]
        # totalstep = cmd.cmd.keywords['totalsteps'].values[0]

        visit = self.actor.visitor.setOrGetVisit(cmd)

        forceMoveArg = 'forceMove' in cmdKeys
        if forceMoveArg is True:
            forceMove = True
        else:
            forceMove = False

        slowOnlyArg = 'slowOnly' in cmdKeys
        if slowOnlyArg is True:
            slowOnly = True
        else:
            slowOnly = False

        # limitOnTime=0.08

        delta = 0.1

        # Switch from default no centroids to default do centroids
        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        # print(self.goodIdx)
        if phi is True:
            eng.setPhiMode()
            steps = stepsize
            day = time.strftime('%Y-%m-%d')
            totalSteps = cmdKeys['totalsteps'].values[0] if 'totalsteps' in cmdKeys else 6000

            self.logger.info(f'Running PHI SLOW motor map.')
            newXml = f'{day}-phi-slow.xml'
            runDir, bad = eng.makePhiMotorMaps(
                newXml, steps=steps, totalSteps=totalSteps, repeat=repeat, fast=False)

            #self.xml = pathlib.Path(f'{runDir}/output/{newXml}')
            #self.cc.pfi.loadModel([self.xml])

            if slowOnly is False:
                self.logger.info(f'Running PHI Fast motor map.')
                newXml = f'{day}-phi-final.xml'
                runDir, bad = eng.makePhiMotorMaps(
                    newXml, steps=steps, totalSteps=totalSteps, repeat=repeat, fast=True)

        else:
            eng.setThetaMode()
            steps = stepsize
            day = time.strftime('%Y-%m-%d')

            if ('totalsteps' in cmdKeys) is False:
                totalstep = 10000
            else:
                totalstep = cmd.cmd.keywords['totalsteps'].values[0]

            self.logger.info(f'Running THETA SLOW motor map.')
            newXml = f'{day}-theta-slow.xml'
            runDir, bad = eng.makeThetaMotorMaps(
                newXml, totalSteps=totalstep, repeat=repeat, steps=steps, delta=delta, fast=False, force=forceMove)

            self.xml = pathlib.Path(f'{runDir}/output/{newXml}')
            self.cc.pfi.loadModel([self.xml])

            if slowOnly is False:
                self.logger.info(f'Running THETA FAST motor map.')
                newXml = f'{day}-theta-final.xml'
                runDir, bad = eng.makeThetaMotorMaps(
                    newXml, totalSteps=totalstep, repeat=repeat, steps=steps, delta=delta, fast=True, force=forceMove)

        cmd.finish(f'Motor map sequence finished')

    def _collectVersions(self):
        """Collect package version"""
        versions = collectVersions(['ics_cobraCharmer', 'ics_fpsActor', 'pfs_instdata', 'pfs_utils', 'datamodel'])
        try:
            mcsVersion = self.actor.models['mcs'].keyVarDict['version'].getValue()
        except ValueError:
            mcsVersion = None

        versions['ics_mcsActor'] = mcsVersion
        versions['moduleXml'] = self.xml
        versions['author'] = "fps"

        return versions

    def _makeDesignName(self, flavour, maskFile):
        """Construct pfsDesign name."""
        if not maskFile:
            return flavour

        _, maskFileName = os.path.split(os.path.splitext(maskFile)[0])
        return f'{flavour}-{maskFileName}'

    def _createHomeDesign(self, cmd):
        """Constructing home pfsDesign."""
        cmdKeys = cmd.cmd.keywords

        if 'theta' in cmdKeys:
            homingType = 'thetaHome'
        elif 'phi' in cmdKeys:
            homingType = 'phiHome'
        else:
            homingType = 'cobraHome'

        thetaEnable = homingType != 'phiHome'
        phiEnable = homingType != 'thetaHome'

        # Getting the index of commanded cobras.
        maskFile = cmdKeys['maskFile'].values[0] if 'maskFile' in cmdKeys else None
        movingIdx = self.loadGoodIdx(maskFile)

        # Making or getting pfsDesign name.
        if 'designName' in cmdKeys:
            designName = cmdKeys['designName'].values[0]
        else:
            designName = self._makeDesignName(homingType, maskFile)

        thetaHome = targetValidation.thetaRange(self.cc.calibModel)
        phiHome = np.zeros_like(thetaHome)

        thetaAngles = thetaHome if thetaEnable else self.atThetas
        phiAngles = phiHome if phiEnable else self.atPhis

        if thetaAngles is None or phiAngles is None:
            cmd.fail('text="Cannot create HOME PfsDesign without cobra information.  Please go to home."')

        positions = self.cc.pfi.anglesToPositions(self.cc.allCobras, thetaAngles, phiAngles)
        versions = self._collectVersions()

        return pfsDesignUtils.createHomeDesign(self.cc.calibModel, positions, movingIdx,
                                               designName=designName,
                                               versions=versions)

    def createHomeDesign(self, cmd):
        """Generating and writing home pfsDesign."""
        pfsDesign = self._createHomeDesign(cmd)

        doWrite, fullPath = pfsDesignUtils.writeDesign(pfsDesign)
        if doWrite:
            cmd.inform(f'text="wrote {fullPath} to disk !"')

        cmd.finish(f'fpsDesignId=0x{pfsDesign.pfsDesignId:016x}')

    def createBlackDotDesign(self, cmd):
        """Generating and writing blackDots pfsDesign."""
        cmdKeys = cmd.cmd.keywords

        # Getting the index of commanded cobras.
        maskFile = cmdKeys['maskFile'].values[0] if 'maskFile' in cmdKeys else None
        movingIdx = self.loadGoodIdx(maskFile)

        # Making or getting pfsDesign name.
        if 'designName' in cmdKeys:
            designName = cmdKeys['designName'].values[0]
        else:
            designName = self._makeDesignName('blackDots', maskFile)

        versions = self._collectVersions()
        pfsDesign = pfsDesignUtils.createBlackDotDesign(self.cc.calibModel, movingIdx,
                                                        designName=designName,
                                                        versions=versions)
        doWrite, fullPath = pfsDesignUtils.writeDesign(pfsDesign)
        if doWrite:
            cmd.inform(f'text="wrote {fullPath} to disk !"')

        cmd.finish(f'fpsDesignId=0x{pfsDesign.pfsDesignId:016x}')

    def createDotConvergenceDesign(self, cmd):
        """Generate and write a pfsDesign targeting the dot-convergence starting positions.

        All cobras in movingIdx are set as BLACKSPOT targets with theta fixed at the
        dot-crossing angle and phi at the ramp start position (computed from
        dotGeometry.computePhiRamp). iic then calls moveToPfsDesign with this design,
        which handles the phi ramp + blind moves internally.
        """
        cmdKeys = cmd.cmd.keywords
        maskFile = cmdKeys['maskFile'].values[0] if 'maskFile' in cmdKeys else None
        movingIdx = self.loadGoodIdx(maskFile)

        if 'designName' in cmdKeys:
            designName = cmdKeys['designName'].values[0]
        else:
            designName = self._makeDesignName('dotConvergence', maskFile)

        thetaDot, _phiCenter, _phiMin, _phiMax, _phiEnter, direction, _halfDot = \
            dotGeometry.computeDotAngles(self.cc)
        _phiInDot = dotGeometry.computePhiAtFraction(_phiCenter, _halfDot, direction,
                                                     dotGeometry.RAMP_LANDING_FRACTION)
        phiStart  = dotGeometry.computePhiStart(_phiInDot, direction)

        versions = self._collectVersions()
        pfsDesign = pfsDesignUtils.createDotConvergenceDesign(
            self.cc.calibModel, self.cc.pfi, self.cc.allCobras,
            thetaDot, phiStart, movingIdx,
            designName=designName, versions=versions)

        doWrite, fullPath = pfsDesignUtils.writeDesign(pfsDesign)
        if doWrite:
            cmd.inform(f'text="wrote {fullPath} to disk !"')

        cmd.finish(f'fpsDesignId=0x{pfsDesign.pfsDesignId:016x}')

    def createThetaPhiScanDesign(self, cmd):
        cmdKeys = cmd.cmd.keywords

        thetaAngle = cmdKeys['thetaAngle'].values[0]
        phiAngle = cmdKeys['phiAngle'].values[0]

        # Making or getting pfsDesign name.
        if 'designName' in cmdKeys:
            designName = cmdKeys['designName'].values[0]
        else:
            designName = f'thetaPhiScan_{thetaAngle:03d}_{phiAngle:03d}'

        cobras = self.cc.allCobras
        thetaAngles = (np.deg2rad(thetaAngle) - self.cc.pfi.calibModel.tht0) % (2 * np.pi)
        phiAngles = np.full_like(thetaAngles, np.deg2rad(phiAngle))

        positions = self.cc.pfi.anglesToPositions(cobras, thetaAngles, phiAngles)
        xy = np.vstack((np.real(positions), np.imag(positions))).T
        versions = self._collectVersions()

        pfsDesign = pfsDesignUtils.createPfsDesign(self.cc.calibModel, xy,
                                                   moveTargetType=TargetType.SCIENCE,
                                                   designName=designName,
                                                   versions=versions)

        doWrite, fullPath = pfsDesignUtils.writeDesign(pfsDesign)
        if doWrite:
            cmd.inform(f'text="wrote {fullPath} to disk !"')

        cmd.finish(f'fpsDesignId=0x{pfsDesign.pfsDesignId:016x}')

    def _exposeMcsAndUpdateAngles(self, cmd, expTime):
        """Take an MCS exposure and update the current cobra angles from measured positions.

        Sets the cobraCoach exposure time, triggers an MCS exposure via
        exposeAndExtractPositions (which also populates cobra_match in opdb),
        then converts the returned PFI positions to theta/phi angles via
        inverse kinematics. Updates both self.atThetas/self.atPhis and
        cobraCharmer's internal angle state via setCurrentAngles, so that
        any subsequent move starts from the correct measured position.

        Parameters
        ----------
        cmd : `actorcore.Command`
            Current command, used for logging.
        expTime : `float`
            MCS exposure time in seconds.
        """
        self.cc.expTime = expTime
        positions = self.cc.exposeAndExtractPositions()

        thetas, phis, _ = self.cc.pfi.positionsToAngles(self.cc.allCobras, positions)
        self.atThetas = thetas[:, 0]
        self.atPhis = phis[:, 0]

        self.cc.setCurrentAngles(self.cc.allCobras, thetaAngles=self.atThetas, phiAngles=self.atPhis)

    def _buildPfsConfigFromMcs(self, cmd, visit, designId):
        """Build, finalize and ingest a pfsConfig using the latest MCS cobra_match data.

        Reads the pfsDesign, assembles the base pfsConfig from gen2 keys, then
        calls the standard finalize/write/ingest pipeline. Intended to be called
        after an MCS exposure has been taken and cobra_match has been populated.

        Parameters
        ----------
        cmd : `actorcore.Command`
            Current command, used for keyword replies and logging.
        visit : `int`
            Current visit id, used to retrieve cobra_match data.
        designId : `int`
            pfsDesign identifier.
        """
        pfsDesign = pfsDesignUtils.readDesign(designId)
        pfsConfig = self.getPfsConfig(cmd, visit=visit, pfsDesign=pfsDesign)

        # A snapshot commands nothing, which is a fact worth recording rather than leaving
        # every cobra to read as "never asked".
        self._finalizeWriteIngestPfsConfig(pfsConfig, cmd=cmd,
                                           cobraCommand=pfsConfigUtils.cobraCommand())

        return pfsConfig

    def genPfsConfigFromMcs(self, cmd):
        cmdKeys = cmd.cmd.keywords

        designId = cmdKeys['designId'].values[0]
        expTime = cmdKeys['expTime'].values[0] if 'expTime' in cmdKeys else None
        visit = self.actor.visitor.setOrGetVisit(cmd)

        self._exposeMcsAndUpdateAngles(cmd, expTime)
        pfsConfig = self._buildPfsConfigFromMcs(cmd, visit=visit, designId=designId)
        cmd.finish(f'text="{pfsConfig.filename} successfully generated..."')

    def moveToHome(self, cmd):
        cmdKeys = cmd.cmd.keywords

        start = time.time()
        convergenceFailed = False
        pfsConfig = None

        expTime = cmdKeys['expTime'].values[0] if 'expTime' in cmdKeys else None
        maskFile = cmdKeys['maskFile'].values[0] if 'maskFile' in cmdKeys else None
        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys
        noMCSexposure = 'noMCSexposure' in cmdKeys
        useMCS = not noMCSexposure
        designId = cmdKeys['designId'].values[0] if 'designId' in cmdKeys else None
        thetaCCW = 'thetaCCW' in cmdKeys

        self.cc.expTime = expTime
        cmd.inform(f'text="Setting moveToHome expTime={expTime}, noMCSexposure={noMCSexposure}"')

        # create or load design.
        if designId:
            cmd.inform(f'text="Reading Design = {designId}"')
            pfsDesign = pfsDesignUtils.readDesign(designId)
            cobraIndex = pfsDesignUtils.homeMaskFromDesign(pfsDesign)
            goodIdx = self.cc.goodIdx[np.isin(self.cc.goodIdx, cobraIndex)]
        else:
            cmd.inform(f'text="maskFile = {maskFile}"')
            # loading mask file and moving only cobra with bitMask==1
            goodIdx = self.loadGoodIdx(maskFile)
            pfsDesign = self._createHomeDesign(cmd)

        cmd.inform(f'text="Getting all avaliable cobra arms."')
        goodCobra = self.cc.allCobras[goodIdx]

        # Homing commands a subset, so the pfsConfig has to say which cobras it drove for
        # the same reason a convergence does.  It validates no targets, so nothing was
        # refused: zero, not NOT_SET, which would claim the file predates the column.
        homed = np.isin(np.arange(self.cc.nCobras), goodIdx)
        command = pfsConfigUtils.cobraCommand(home=homed)
        validationMask = np.zeros(self.cc.nCobras, dtype=np.int32)

        # Only grab a visit if we need one for the PFSC and pfsConfig files
        visit = self.actor.visitor.setOrGetVisit(cmd) if (useMCS or 'visit' in cmdKeys) else -1

        # Making base pfsConfig from design file, fetching additional keys from gen2.
        pfsConfig = self.getPfsConfig(cmd, visit=visit, pfsDesign=pfsDesign)

        # Deactivating both theta and phi.
        thetaEnable = phiEnable = False

        if phi:
            eng.setPhiMode()
            phiEnable = True

        elif theta:
            eng.setThetaMode()
            thetaEnable = True

        else:
            eng.setNormalMode()
            thetaEnable = phiEnable = True

        # Invalidating previous pfsConfig.
        cmd.inform(f'pfsConfig=0x{pfsDesign.pfsDesignId:016x},{visit},inProgress')

        try:
            diff = self.cc.moveToHome(goodCobra, thetaEnable=thetaEnable, phiEnable=phiEnable,
                                  thetaCCW=thetaCCW, noMCS=noMCSexposure)

            if useMCS and thetaEnable and phiEnable and diff is not None:
                self.logger.info(f'Averaged position offset compared with cobra center = {np.mean(diff)}')

            thetaHome = targetValidation.thetaRange(self.cc.calibModel)

            if self.atThetas is None:
                self.atThetas = thetaHome.copy()
            if self.atPhis is None:
                self.atPhis = np.zeros(len(self.cc.allCobras))

            if phiEnable:
                self.atPhis[goodIdx] = 0
                cmd.inform(f'text="Setting phiAngle to the home position."')

            if thetaEnable:
                self.atThetas[goodIdx] = thetaHome[goodIdx]
                cmd.inform(f'text="Setting thetaAngle to the home position."')

            self.cc.setCurrentAngles(goodCobra, thetaAngles=self.atThetas[goodIdx], phiAngles=self.atPhis[goodIdx])

        except Exception:
            convergenceFailed = True
            raise
        finally:
            eng.setNormalMode()
            # Only generate pfsConfigs if we take an image which needs them.
            if visit !=-1:
                pfsConfig.convergenceParams = dict(elapsedTime=round(time.time() - start, 3))
                self._finalizeWriteIngestPfsConfig(pfsConfig, cmd=cmd,
                                                   convergenceFailed=convergenceFailed,
                                                   validationMask=validationMask,
                                                   cobraCommand=command)

        cmd.finish(f'text="Moved all arms back to home"')

    def cobraAndDotRecenter(self, cmd):
        """
            Making a new XML using home position instead of rotational center
        """
        visit = self.actor.visitor.setOrGetVisit(cmd)

        daytag = time.strftime('%Y%m%d')
        newXml = eng.convertXML2(f'recenter_{daytag}.xml', homePhi=False)

        self.logger.info(f'Using new XML = {newXml} as default setting')
        self.xml = newXml

        self.cc.calibModel = pfiDesign.PFIDesign(pathlib.Path(self.xml))
        cmd.inform(f'text="Loading new XML file= {newXml}"')

        self.logger.info(f'Loading conversion matrix for {self.cc.frameNum}')
        frameNum = self.cc.frameNum
        # Use this latest matrix as initial guess for automatic calculating.
        db = self.connectToDB(cmd)
        sql = f'''SELECT * from mcs_pfi_transformation 
            WHERE mcs_frame_id < {frameNum} ORDER BY mcs_frame_id DESC
            FETCH FIRST ROW ONLY
            '''
        transMatrix = db.fetch_query(sql)
        scale = transMatrix['x_scale'].values[0]
        xoffset = transMatrix['x_trans'].values[0]
        yoffset = transMatrix['y_trans'].values[0]
        # Always
        angle = -transMatrix['angle'].values[0]
        self.logger.info(f'Latest matrix = {xoffset} {yoffset} scale = {scale}, angle={angle}')

        # Loading FF from DB
        ff_f3c = self.nv.readFFConfig()['x'].values + self.nv.readFFConfig()['y'].values * 1j
        rx, ry = fpstool.projectFCtoPixel([ff_f3c.real, ff_f3c.imag], scale, angle, [xoffset, yoffset])

        # Load MCS data from DB
        self.logger.info(f'Load frame from DB')
        mcsData = self.nv.readCentroid(frameNum)

        target = np.array([rx, ry]).T.reshape((len(rx), 2))
        source = np.array([mcsData['centroidx'].values, mcsData['centroidy'].values]
                          ).T.reshape((len(mcsData['centroidx'].values), 2))

        match = fpstool.pointMatch(target, source)
        ff_mcs = match[:, 0] + match[:, 1] * 1j

        self.logger.info(f'Mapping DOT location using latest affine matrix')

        # afCoeff = cal.tranformAffine(ffpos, ff_mcs)
        ori = np.array([np.array([self.cc.calibModel.ffpos.real, self.cc.calibModel.ffpos.imag]).T])
        tar = np.array([np.array([ff_mcs.real, ff_mcs.imag]).T])
        # self.logger.info(f'{ori}')
        # self.logger.info(f'{tar}')
        afCoeff, inlier = cv2.estimateAffinePartial2D(np.array(ori), np.array(tar))

        afCor = cv2.transform(np.array(
            [np.array([self.cc.calibModel.dotpos.real, self.cc.calibModel.dotpos.imag]).T]), afCoeff)
        newDotPos = afCor[0]
        self.cc.calibModel.dotpos = newDotPos[:, 0] + newDotPos[:, 1] * 1j

        cmd.finish(f'text="New XML file {newXml} is generated."')

    def targetConverge(self, cmd):
        """ Making target convergence test. """
        cmdKeys = cmd.cmd.keywords
        runs = cmd.cmd.keywords['totalTargets'].values[0]
        maxsteps = cmd.cmd.keywords['maxsteps'].values[0]
        ontime = 'ontime' in cmdKeys
        speed = 'speed' in cmdKeys

        visit = self.actor.visitor.setOrGetVisit(cmd)

        eng.setNormalMode()
        self.logger.info(f'Moving cobra to home position')
        self.cc.moveToHome(self.cc.goodCobras, thetaEnable=True, phiEnable=True, thetaCCW=False)

        self.logger.info(f'Making transformation using home position')
        daytag = time.strftime('%Y%m%d')

        # We don't need to home phi again since there is a home sequence above.
        newXml = eng.convertXML2(f'{daytag}.xml', homePhi=False)

        self.logger.info(f'Using new XML = {newXml} as default setting')
        self.xml = newXml

        self.cc.loadModel(file=pathlib.Path(self.xml))
        # eng.setCobraCoach(self.cc)

        if ontime is True:
            self.logger.info(f'Run convergence test of {runs} targets with constant on-time')
            self.logger.info(f'Setting max step = {maxsteps}')
            eng.setConstantOntimeMode(maxSteps=maxsteps)

            targets, moves = eng.convergenceTest2(self.cc.goodIdx, runs=runs, thetaMargin=np.deg2rad(15.0),
                                                  phiMargin=np.deg2rad(15.0), thetaOffset=0,
                                                  phiAngle=(np.pi * 5 / 6, np.pi / 3, np.pi / 4),
                                                  tries=16, tolerance=0.2, threshold=20.0,
                                                  newDir=True, twoSteps=False)

        if speed is True:
            self.logger.info(f'Run convergence test of {runs} targets with constant speed')
            self.logger.info(f'Setting max step = {maxsteps}')

            mmTheta = np.load('/data/MCS/20210505_016/data/thetaOntimeMap.npy')
            mmThetaSlow = np.load('/data/MCS/20210505_017/data/thetaOntimeMap.npy')
            mmPhi = np.load('/data/MCS/20210506_013/data/phiOntimeMap.npy')
            mmPhiSlow = np.load('/data/MCS/20210506_014/data/phiOntimeMap.npy')

            self.logger.info(f'On-time maps loaded.')

            mmTheta = self._cleanAnomaly(mmTheta)
            mmThetaSlow = self._cleanAnomaly(mmThetaSlow)
            mmPhi = self._cleanAnomaly(mmPhi)
            mmPhiSlow = self._cleanAnomaly(mmPhiSlow)

            eng.setConstantSpeedMaps(mmTheta, mmPhi, mmThetaSlow, mmPhiSlow)

            eng.setConstantSpeedMode(maxSegments=int({maxsteps} / 100), maxSteps=100)

            self.logger.info(f'Setting maxstep = 100, nSeg = {int({maxsteps} / 100)}')
            targets, moves = eng.convergenceTest2(self.cc.goodIdx, runs=runs,
                                                  thetaMargin=np.deg2rad(15.0), phiMargin=np.deg2rad(15.0),
                                                  thetaOffset=0, phiAngle=(np.pi * 5 / 6, np.pi / 3, np.pi / 4),
                                                  tries=16, tolerance=0.2, threshold=20.0, newDir=True, twoSteps=False)

        cmd.finish(f'target convergece is finished')

    def angleConverge(self, cmd):
        """ Making comvergence test for a specific arm. """
        cmdKeys = cmd.cmd.keywords
        runs = cmd.cmd.keywords['angleTargets'].values[0]
        visit = self.actor.visitor.setOrGetVisit(cmd)

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        if phi is True:
            self.logger.info(f'Run phi convergence test of {runs} targets')
            eng.setPhiMode()

            eng.phiConvergenceTest(self.cc.goodIdx, runs=runs, tries=12, fast=False, tolerance=0.1)
            cmd.finish(f'text="angleConverge of phi arm is finished"')
        else:
            self.logger.info(f'Run theta convergence test of {runs} targets')
            # eng.setThetaMode()
            eng.thetaConvergenceTest(self.cc.goodIdx, runs=runs, tries=12, fast=False, tolerance=0.1)
            cmd.finish(f'text="angleConverge of theta arm is finished"')

    def moveToThetaAngleFromOpenPhi(self, cmd):
        """ Move cobras to nominal safe position: thetas OUT, phis in.
        Assumes phi is at 60deg and that we know thetaPositions.
        """
        cmdKeys = cmd.cmd.keywords
        visit = self.actor.visitor.setOrGetVisit(cmd)

        angleList = np.load(f'/data/MCS/20210816_090/output/phiOpenAngle')

        cobraIdx = np.arange(2394)
        thetas = np.full(len(2394), 0.5 * np.pi)
        thetas[cobraIdx < 798] += np.pi * 2 / 3
        thetas[cobraIdx >= 1596] -= np.pi * 2 / 3
        thetas = thetas % (np.pi * 2)

        phiAngle = angleList
        tolerance = np.rad2deg(0.5)
        angle = (180.0 - phiAngle) / 2.0
        thetaAngles = np.full(len(self.allCobras), -angle, dtype='f4')
        thetaAngles[np.arange(0, self.nCobras, 2)] += 0
        thetaAngles[np.arange(1, self.nCobras, 2)] += 180

        dataPath, diffAngles, moves = eng.moveThetaAngles(self.cc.goodIdx, thetaAngles,
                                                          relative=False, local=True, tolerance=0.002, tries=12,
                                                          fast=False, newDir=True)

        cmd.finish(f'text="gotoSafeFromPhi60 is finished"')

    def movePhiForThetaOps(self, cmd):
        """ Move PHI to a certain angle to avoid DOT for theta MM. """
        bigAngle, smallAngle = 75, 30
        cmdKeys = cmd.cmd.keywords
        runDir = pathlib.Path(cmd.cmd.keywords['runDir'].values[0])

        newDot, rDot = fpstool.alignDotOnImage(runDir)

        arm = 'phi'
        centers = np.load(f'{runDir}/data/{arm}Center.npy')
        radius = np.load(f'{runDir}/data/{arm}Radius.npy')
        fw = np.load(f'{runDir}/data/{arm}FW.npy')

        self.logger.info(f'Total cobra arms = {self.cc.nCobras}, try angle {bigAngle}')
        angleList = np.zeros(self.cc.nCobras) + bigAngle
        L1, blockId = fpstool.checkPhiOpenAngle(centers, radius, fw, newDot, rDot, angleList)

        self.logger.info(f'Total {len(blockId)} arms are blocked by DOT, try abgle = {smallAngle} ')

        angleList[blockId] = 30
        L1, blockId = fpstool.checkPhiOpenAngle(centers, radius, fw, newDot, rDot, angleList)
        self.logger.info(f'Total {len(blockId)} arms are blocked by DOT')

        self.logger.info(f'Move phi to requested angle')

        # move phi to 60 degree for theta test
        dataPath, diffAngles, moves = eng.movePhiAngles(self.cc.goodIdx, np.deg2rad(angleList[self.cc.goodIdx]),
                                                        relative=False, local=True, tolerance=0.002,
                                                        tries=12, fast=False, newDir=True)

        self.logger.info(f'Data path : {dataPath}')

        np.save(f'{runDir}/output/phiOpenAngle', angleList)

        cmd.finish(f'text="PHI is opened at requested angle for theta MM operation!"')

    def movePhiToAngle(self, cmd):
        """ Making comvergence test for a specific arm. """
        cmdKeys = cmd.cmd.keywords
        angle = cmd.cmd.keywords['angle'].values[0]
        itr = cmd.cmd.keywords['iteration'].values[0]
        visit = self.actor.visitor.setOrGetVisit(cmd)

        if itr == 0:
            itr = 8

        self.logger.info(f'Move phi to angle = {angle}')

        # move phi to 60 degree for theta test
        dataPath, diffAngles, moves = eng.movePhiAngles(self.cc.goodIdx, np.deg2rad(angle),
                                                        relative=False, local=True, tolerance=0.002,
                                                        tries=itr, fast=False, newDir=True)

        self.logger.info(f'Data path : {dataPath}')
        cmd.finish(f'text="PHI is now at {angle} degrees!"')

    def movePhiForDots(self, cmd):
        """ Making a convergence test to a specified phi angle. """
        cmdKeys = cmd.cmd.keywords
        angle = cmd.cmd.keywords['angle'].values[0]
        itr = cmd.cmd.keywords['iteration'].values[0]
        visit = self.actor.visitor.setOrGetVisit(cmd)

        if itr == 0:
            itr = 8

        self.logger.info(f'Move phi to angle = {angle}')

        # move phi to certain degree for theta test
        eng.moveToPhiAngleForDot(self.cc.goodIdx, angle, tolerance=0.01,
                                 tries=12, homed=False, newDir=False, threshold=2.0, thetaMargin=np.deg2rad(15.0))

        cmd.finish(f'text="PHI is now at {angle} degrees!"')

    def moveToSafePosition(self, cmd):

        """ Move cobras to nominal safe position: thetas OUT, phis in.
        Assumes phi is at 60deg and that we know thetaPositions.
        """
        cmdKeys = cmd.cmd.keywords
        visit = self.actor.visitor.setOrGetVisit(cmd)
        expTime = cmdKeys['expTime'].values[0] if 'expTime' in cmdKeys else None
        tolerance = cmdKeys['tolerance'].values[0] if 'tolerance' in cmdKeys else 0.1
        phiAngle = cmdKeys['phiAngle'].values[0] if 'phiAngle' in cmdKeys else 80
        thetaAngle = cmdKeys['thetaAngle'].values[0] if 'thetaAngle' in cmdKeys else 60
        goHome = 'noHome' not in cmdKeys

        thetas = np.full(len(self.cc.goodIdx), np.deg2rad(thetaAngle))
        phis = np.full(len(self.cc.goodIdx), np.deg2rad(phiAngle))

        cobras = self.cc.allCobras[self.cc.goodIdx]
        targets = self.cc.pfi.anglesToPositions(cobras, thetas, phis)

        self.cc.pfi.resetMotorScaling()
        # "homed" should be "goHome". Hack now here, fix there later.
        dataPath, thetas, phis, moves = eng.moveThetaPhi(self.cc.goodIdx, thetas, phis,
                                                         False, False, tolerance=tolerance,
                                                         tries=8, homed=goHome, newDir=True,
                                                         threshold=2.0, thetaMargin=np.deg2rad(15.0))

        # Save the moves for record.
        np.save(dataPath / 'targets', targets)
        np.save(dataPath / 'moves', moves)
        np.save(dataPath / 'thetas', thetas)
        np.save(dataPath / 'phis', phis)
        cmd.inform(f'text="Data of moves are saved to {dataPath}"')

        cmd.finish(f'text="moveToSafePosition is finished"')

    def motorOntimeSearch(self, cmd):
        """ FPS interface of searching the on time parameters for a specified motor speed """
        cmdKeys = cmd.cmd.keywords
        visit = self.actor.visitor.setOrGetVisit(cmd, doAutoVisit=True)

        # self._connect()

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        if phi is True:
            day = time.strftime('%Y-%m-%d')
            newXml = f'{day}-phi_opt.xml'

            xml = eng.phiOnTimeSearch(newXml, speeds=(0.06, 0.12), steps=(500, 250), iteration=3, repeat=1)

            cmd.finish(f'text="motorOntimeSearch of phi arm is finished"')
        else:
            day = time.strftime('%Y-%m-%d')
            newXml = f'{day}-theta_opt.xml'
            xml = eng.thetaOnTimeSearch(newXml, speeds=(0.06, 0.12), steps=[1000, 500], iteration=3, repeat=1)
            self.logger.info(f'Theta on-time optimal XML = {xml}')
            cmd.finish(f'text="motorOntimeSearch of theta arm is finished"')

    def makeOntimeMap(self, cmd):
        """ Making on-time map. """
        cmdKeys = cmd.cmd.keywords
        visit = self.actor.visitor.setOrGetVisit(cmd, doAutoVisit=True)

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        if phi is True:

            self.logger.info(f'Running phi fast on-time scan.')
            dataPath, ontimes, angles, speeds = eng.phiOntimeScan(speed=np.deg2rad(0.12),
                                                                  steps=10, totalSteps=6000, repeat=1, scaling=4.0)

            self.logger.info(f'Running phi slow on-time scan.')
            dataPath, ontimes, angles, speeds = eng.phiOntimeScan(speed=np.deg2rad(0.06),
                                                                  steps=20, totalSteps=9000, repeat=1, scaling=4.0)
        else:
            self.logger.info(f'Running theta fast on-time scan.')
            dataPath, ontimes, angles, speeds = eng.thetaOntimeScan(speed=np.deg2rad(0.12), steps=10,
                                                                    totalSteps=10000, repeat=1, scaling=3.0)

            self.logger.info(f'Running theta slow on-time scan.')
            dataPath, ontimes, angles, speeds = eng.thetaOntimeScan(speed=np.deg2rad(0.06), steps=20,
                                                                    totalSteps=15000, repeat=1, scaling=3.0,
                                                                    tolerance=np.deg2rad(3.0))

        cmd.finish(f'text="Motor on-time scan is finished."')

    def loadGoodIdx(self, maskFile):
        """Return cobraIndex that is flagged to be moved."""

        def loadMaskFile():
            """Just return cobra index where bitMask==1"""
            df = pd.read_csv(maskFile, index_col=0)
            doMoveCobraIds = df[df.bitMask.astype('bool')].cobraId.to_numpy()
            return doMoveCobraIds - 1

        # self.logger.info(f"loadGoodIdx maskfile = {maskFile}")

        # doMove = self.cc.goodIdx if maskFile is None else loadMaskFile()
        # doMove = self.cc.goodIdx if maskFile is None else self.logger.info(f"loadGoodIdx maskfile = {maskFile}")
        if maskFile is None:
            doMove = self.cc.goodIdx
        else:
            doMove = loadMaskFile()

        return self.cc.goodIdx[np.isin(self.cc.goodIdx, doMove)]

    def getPfsConfig(self, cmd, visit, pfsDesign):
        """Get pfsConfig from pfsDesign, adding additional gen2 keys."""
        cards = fits.getPfsConfigCards(self.actor, cmd, visit, expType='acquisition')
        versions = self._collectVersions()
        return pfsConfigUtils.pfsConfigFromDesign(pfsDesign, visit,
                                                  calibModel=self.cc.calibModel,
                                                  header=cards,
                                                  versions=versions)

    def moveToPfsDesign(self, cmd):
        """ Move cobras to a PFS design. """
        thetaMarginDeg = 15.0

        """
        Initialize the cobra control parameters. When moving to a design, these parameters need to be set.
        If the convergence sequence is not completed, the parameters may continue to increase.
        Therefore, it is necessary to reset the parameters.
        """
        self.cc.useScaling = False
        self.cc.maxSegments = 10
        self.cc.maxTotalSteps = 2000

        start = time.time()
        convergenceFailed = False
        cmdKeys = cmd.cmd.keywords

        designId = cmdKeys['designId'].values[0]
        expTime = cmdKeys['expTime'].values[0] if 'expTime' in cmdKeys else None
        maskFile = cmdKeys['maskFile'].values[0] if 'maskFile' in cmdKeys else None
        iteration = cmdKeys['iteration'].values[0] if 'iteration' in cmdKeys else 12
        tolerance = cmdKeys['tolerance'].values[0] if 'tolerance' in cmdKeys else 0.01
        fastThreshold = cmdKeys['fastThreshold'].values[0] if 'fastThreshold' in cmdKeys else 99.9

        convergenceCfg = self.actor.actorConfig.get('convergence', {})
        maxInvalidScienceTargets = convergenceCfg.get('maxInvalidScienceTargets', 25)
        targetFallback = pfsConfigUtils.resolveTargetFallback(convergenceCfg)
        notConvergedDistanceThreshold = convergenceCfg.get('notConvergedDistanceThreshold', 0.05)
        notConvergedDistanceThreshold = max(notConvergedDistanceThreshold, 5 * tolerance)

        shortExp = 'shortExpOff' not in cmdKeys
        twoSteps = 'twoStepsOff' not in cmdKeys
        goHome = 'goHome' in cmdKeys
        doTweak = 'noTweak' not in cmdKeys
        skipFiducialInterferenceCheck = 'skipFiducialInterferenceCheck' in cmdKeys
        # Proceed even when more science targets fail validation than the limit.
        allowMaskedCobras = 'allowMaskedCobras' in cmdKeys

        self.cc.expTime = expTime
        cmd.inform(f'text="Setting moveToPfsDesign expTime={expTime}"')
        cmd.inform(f'text="Running moveToPfsDesign with tolerance={tolerance} iteration={iteration} "')
        cmd.inform(f'text="moveToPfsDesign with twoSteps={twoSteps} goHome={goHome}"')

        visit = self.actor.visitor.setOrGetVisit(cmd)

        # Loading pfsDesign file.
        pfsDesign = pfsDesignUtils.readDesign(designId)

        # making base pfsConfig from design file, fetching additional keys from gen2.
        pfsConfig = self.getPfsConfig(cmd, visit=visit, pfsDesign=pfsDesign)
        cmd.inform(f'pfsConfig=0x{designId:016x},{visit},Preparing')

        if doTweak:
            # Last minute tweaking for proper motion / parallax ..
            cmd.inform(f'text="Tweaking designed targets position..."')
            tweakTargetPosition(pfsConfig)

        # ── validation ────────────────────────────────────────────────────────
        # Everything here is (nCobras,) and indexed by cobra index; subsetting is done
        # once, at the end, when the arrays are handed to the move.
        targets = pfsConfigUtils.makeTargetsArray(pfsConfig)
        isNan = np.isnan(targets)

        isScience = pfsConfigUtils.getCobraTargetMask(pfsConfig, pfsConfigUtils.SCIENCE_TARGET_TYPES)
        isBlackDot = pfsConfigUtils.getCobraTargetMask(pfsConfig, [TargetType.BLACKSPOT])
        isUnassigned = pfsConfigUtils.getCobraTargetMask(pfsConfig, [TargetType.UNASSIGNED])

        # A cobra carrying DCB, AFL, SUNSS or HOME means the design was built for another
        # purpose.  Refuse it: moving the remainder would half-execute somebody else's
        # design, and the caller cannot tell that from a normal convergence.
        accepted = pfsConfigUtils.getCobraTargetMask(pfsConfig, pfsConfigUtils.ACCEPTED_TARGET_TYPES)
        if not accepted.all():
            cmd.fail(f'text="{(~accepted).sum()} cobras carry a forbidden target type for convergence"')
            return

        flags = targetValidation.validateTargets(self.cc.calibModel, targets,
                                                 skipFiducialInterferenceCheck=skipFiducialInterferenceCheck)
        isInvalid = targetValidation.isInvalid(flags)

        # ── outcomes ──────────────────────────────────────────────────────────
        # Before the alert: a cobra that will never be commanded cannot make a design out
        # of spec, and 36 of the 42 broken ones read out of reach on any target.
        thetaRangeAll = targetValidation.thetaRange(self.cc.calibModel)
        mayMove = np.isin(np.arange(self.cc.nCobras), self.loadGoodIdx(maskFile))
        isBroken = np.zeros(self.cc.nCobras, dtype=bool)
        isBroken[self.cc.badIdx] = True

        # An isInvalid target is parked on its black dot rather than left wherever it
        # happens to be.  Drop `| isInvalid` here to go back to only explicit BLACKSPOT.
        toBlackDot = (isBlackDot | isUnassigned | isInvalid) & mayMove
        converge = isScience & ~toBlackDot & mayMove

        # Before the gate, so a refused design still reports why it was refused.
        for diag in pfsConfigUtils.summariseByTargetType(pfsConfig, flags, converge, toBlackDot,
                                                         isBroken, ~mayMove & ~isBroken):
            cmd.inform(f'text="{diag}"')

        nBad = int((isInvalid & isScience & mayMove).sum())
        if nBad > maxInvalidScienceTargets and not allowMaskedCobras:
            cmd.fail(f'text="{nBad} invalid science targets, above '
                     f'maxInvalidScienceTargets={maxInvalidScienceTargets}; design is out of spec"')
            return

        thetaSolution, phiSolution, _ = self.cc.pfi.positionsToAngles(self.cc.allCobras, targets)
        thetas, phis = thetaSolution[:, 0], phiSolution[:, 0]

        # Bump theta targets near the CW hard stop to their equivalent above 360.  The
        # margin is half the hard-stop overlap, so the result stays in [0, thetaRange].
        optMargin = (thetaRangeAll - 2 * np.pi) / 2
        thetas[thetas < optMargin] += np.pi * 2

        # Dot cobras follow a phi ramp into their black dot rather than going to the
        # design position.  buildSafeRamp returns (nCobras,) and (nIter, nCobras).
        dotGlobalIdx = np.flatnonzero(toBlackDot)
        thetaDot, phiStart, phiRamp, thetaRamp, dotGeom = dotGeometry.buildSafeRamp(
            self.cc, dotGlobalIdx, iteration)
        thetas[toBlackDot] = thetaDot[toBlackDot]
        phis[toBlackDot] = phiStart[toBlackDot]

        # ── subset, once, here ────────────────────────────────────────────────
        # Above is (nCobras,); below, `commanded` is the mask and `commanded*` is an array
        # restricted to it, so the prefix alone states an array's width.  commandedIdx
        # exists only because the engineer API takes cobra ids where a mask would do.
        commanded = converge | toBlackDot
        commandedIdx = np.flatnonzero(commanded)
        commandedThetas = thetas[commanded]
        commandedPhis = phis[commanded]
        commandedTargets = targets[commanded]
        commandedToDot = toBlackDot[commanded]
        commandedPhiRamp = phiRamp[:, commanded]
        commandedThetaRamp = thetaRamp[:, commanded]

        # Drives FiberStatus in finalize(): a science fiber parked on a black dot is
        # commanded but not good, or the DRP extracts a deliberately dark fiber.
        validationMask = pfsConfigUtils.targetValidationMask(flags, isScience)
        command = pfsConfigUtils.cobraCommand(converge=converge, toBlackDot=toBlackDot)
        pfsConfig.convergenceParams = pfsConfigUtils.convergenceParams(
            targetFallback, notConvergedDistanceThreshold, skipFiducialInterferenceCheck,
            numIterations=iteration, requestedTolerance=tolerance)

        # Nothing to command means the design is not one this command can act on -- every
        # target type unaccepted, or every cobra masked.  Say so: the engineer API takes
        # cobra ids, and an empty index array reaches numpy as float64 and raises
        # "arrays used as indices must be of integer type" several frames down.
        if not commanded.any():
            cmd.fail('text="no cobra to command: check the target types warned about above"')
            return

        # Here we start to deal with target table
        cmd.inform(f'text="Handling the cobra target table."')
        self.cc.trajectoryMode = True
        traj, moves = eng.createTrajectory(commandedIdx, commandedThetas, commandedPhis,
                                           tries=iteration, twoSteps=True, threshold=fastThreshold, timeStep=500)
        moves[:, 2]['position'] = commandedTargets

        cmd.inform(f'text="Reset the current angles for cobra arms."')
        self.cc.trajectoryMode = False
        thetaHome = targetValidation.thetaRange(self.cc.calibModel)

        if goHome:
            cmd.inform(f'text="Setting ThetaAngle = Home and phiAngle = 0."')
            self.cc.setCurrentAngles(self.cc.allCobras, thetaAngles=thetaHome, phiAngles=0)

            cobraTargetTable = najaVenator.CobraTargetTable(visit, iteration, self.cc.calibModel, designId, goHome=True)

        else:
            # Check if we have atThetas and atPhis. If not, we cannot proceed.
            if self.atThetas is None or self.atPhis is None:
                cmd.fail('text="Cannot move to PfsDesign without cobra information.  Please use goHome option."')
                return

            cmd.inform(f'text="Number of cobras = {len(commandedIdx)} '
                       f'Number of angles = {len(self.atThetas[commandedIdx])}."')
            self.cc.setCurrentAngles(self.cc.allCobras[commandedIdx],
                                     thetaAngles=self.atThetas[commandedIdx],
                                     phiAngles=self.atPhis[commandedIdx])

            cobraTargetTable = najaVenator.CobraTargetTable(visit, iteration, self.cc.calibModel, designId,
                                                            goHome=False)

        # Getting a new directory for this operation by running PFI connection using cobraCoach.
        # This operation will update dataDir for both PFI and camera.  So that we can keep information correctly.
        self.cc.connect(False)

        # Saving information for book keeping. First, getting path name for saving data.
        dataPath = pathlib.Path(self.cc.runManager.dataDir)

        # Save the solution flags for debugging and record.
        np.save(f'{dataPath}/validationFlags', flags)

        # Save the filtered targets for record. These are the targets that we will actually move to.
        np.save(f'{dataPath}/targets', commandedTargets)
        cmd.inform(f'text="Saving targets list to file {dataPath}/targets.npy."')

        # Every exclusion reason comes from the one validation, so the log cannot
        # disagree with the decision that was actually taken.
        filtering_records = []
        for idx in np.flatnonzero(~mayMove):
            filtering_records.append({'cobra_id': idx, 'step': 'mask file',
                                      'reason': 'excluded_by_mask_or_broken'})
        for idx in np.flatnonzero(toBlackDot):
            filtering_records.append({'cobra_id': idx, 'step': 'dot_cobra',
                                      'reason': 'blackspot_target'})
        logged = isInvalid | isUnassigned
        for bit, name in targetValidation.FLAG_NAMES.items():
            if bit == targetValidation.IN_OVERLAPPING_REGION:
                continue        # informational: set on nearly every cobra, not a defect
            for idx in np.flatnonzero(((flags & bit) != 0) & logged):
                filtering_records.append({'cobra_id': idx, 'step': 'validation',
                                          'reason': name.replace(' ', '_')})

        df_log = pd.DataFrame(filtering_records)
        df_log.to_csv(f'{dataPath}/cobra_filtering_log.csv', index=False)

        np.savez(f'{dataPath}/cobra_filtering.npz',
                 validation_flags=flags,
                 nan_targets=isNan,
                 converge=converge,
                 to_black_dot=toBlackDot,
                 do_not_move=~mayMove,
                 validation_mask=validationMask,
                 cobra_command=command,
                 final_moving_cobras=commandedIdx)

        cmd.inform(f'text="Saved filtering data: {len(filtering_records)} exclusions logged"')

        cmd.inform(f'text="Reset the motor scaling factor."')
        self.cc.pfi.resetMotorScaling(self.cc.allCobras)

        # Pre-compute per-iteration commanded target positions from geometry.
        # Done here — before any movement — so MCS can use the table for cobra identification.
        commandedThetaSchedule, commandedPhiSchedule = dotGeometry.buildCommandedAngle(
            commandedThetas, commandedPhis, commandedPhiRamp, commandedThetaRamp)
        commandedTargetPositions = np.zeros((len(commandedIdx), iteration), dtype=complex)
        commandedCobras = self.cc.allCobras[commandedIdx]
        for j in range(iteration):
            commandedTargetPositions[:, j] = self.cc.pfi.anglesToPositions(commandedCobras,
                                                                   commandedThetaSchedule[j],
                                                                   commandedPhiSchedule[j])
        cobraTargetTable.makeTargetTableFromTargetPositions(commandedTargetPositions, self.cc, commandedIdx)
        cobraTargetTable.writeTargetTable()

        # Invalidating previous pfsConfig.
        cmd.inform(f'pfsConfig=0x{pfsDesign.pfsDesignId:016x},{visit},inProgress')

        # Drop any previous visit's scan state up front: if this convergence
        # raises, moveToDotByFlux must not blind-move on stale indices.
        self.dotScan = None

        try:
            if twoSteps:
                moves = np.zeros((1, len(commandedIdx), iteration), dtype=eng.moveDtype)
                commandedThetaRange = thetaRangeAll[commandedIdx]

                # limit phi angle for first two tries
                commandedLimitPhi = np.pi / 3 - self.cc.calibModel.phiIn[commandedIdx] - np.pi
                commandedThetasVia = np.copy(commandedThetas)
                commandedPhisVia = np.copy(commandedPhis)
                for c in range(len(commandedIdx)):
                    if commandedToDot[c]:
                        continue
                    if commandedPhis[c] > commandedLimitPhi[c]:
                        commandedPhisVia[c] = commandedLimitPhi[c]
                        commandedThetasVia[c] = commandedThetas[c] + (commandedPhis[c] - commandedLimitPhi[c]) / 2
                        if commandedThetasVia[c] > commandedThetaRange[c]:
                            commandedThetasVia[c] = commandedThetaRange[c]

                _useScaling, _maxSegments, _maxTotalSteps = self.cc.useScaling, self.cc.maxSegments, self.cc.maxTotalSteps
                self.cc.useScaling, self.cc.maxSegments, self.cc.maxTotalSteps = False, _maxSegments * 2, _maxTotalSteps * 2
                cmd.inform(
                    f'text="useScaling={self.cc.useScaling}, maxSegments={self.cc.maxSegments}, maxTotalSteps={self.cc.maxTotalSteps}"')

                if shortExp is True:
                    cmd.inform(f'text="Using 0.8 second exposure time for first three iteration."')
                    self.cc.expTime = 0.8
                else:
                    cmd.inform(f'text="Using {expTime} second exposure time for first three iteration."')
                    self.cc.expTime = expTime

                cmd.inform(f'text="Cobra goHome is set to be {goHome}"')
                # INSTRM-2976: fast motor map disabled in the two-step convergence
                dataPath, atThetas, atPhis, moves[0, :, :2] = \
                    eng.moveThetaPhi(commandedIdx, commandedThetasVia, commandedPhisVia,
                                     relative=False, local=True, tolerance=tolerance,
                                     tries=2, homed=goHome, newDir=False, thetaFast=False, phiFast=False,
                                     threshold=fastThreshold, thetaMargin=np.deg2rad(thetaMarginDeg),
                                     phiRamp=commandedPhiRamp[:2], thetaRamp=commandedThetaRamp[:2])

                self.cc.expTime = expTime
                self.cc.useScaling, self.cc.maxSegments, self.cc.maxTotalSteps = _useScaling, _maxSegments, _maxTotalSteps
                cmd.inform(
                    f'text="useScaling={self.cc.useScaling}, maxSegments={self.cc.maxSegments}, maxTotalSteps={self.cc.maxTotalSteps}"')

                dataPath, atThetas, atPhis, moves[0, :, 2:] = \
                    eng.moveThetaPhi(commandedIdx, commandedThetas, commandedPhis,
                                     relative=False, local=True,
                                     tolerance=tolerance,
                                     tries=iteration - 2,
                                     homed=False,
                                     newDir=False, thetaFast=False, phiFast=False, threshold=fastThreshold,
                                     thetaMargin=np.deg2rad(thetaMarginDeg),
                                     phiRamp=commandedPhiRamp[2:], thetaRamp=commandedThetaRamp[2:],
                                     )

            else:
                dataPath, atThetas, atPhis, moves = eng.moveThetaPhi(commandedIdx, commandedThetas, commandedPhis,
                                                                     relative=False, local=True, tolerance=tolerance,
                                                                     tries=iteration, homed=goHome, newDir=False,
                                                                     thetaFast=False, phiFast=False,
                                                                     threshold=fastThreshold,
                                                                     thetaMargin=np.deg2rad(thetaMarginDeg),
                                                                     phiRamp=commandedPhiRamp,
                                                                     thetaRamp=commandedThetaRamp,
                                                                     )
            self.atThetas = atThetas
            self.atPhis = atPhis

            # Push the hidden dot cobras deeper inside the dot with a single open-loop
            # (slow-map) move: the closed-loop ramp above lands them at
            # RAMP_LANDING_FRACTION, this takes them to BLIND_TARGET_FRACTION.
            if len(dotGlobalIdx):
                dotGeometry.blindMoveHiddenCobras(self.cc, dotGlobalIdx, commandedIdx,
                                                  moves, dotGeom,
                                                  fromFraction=dotGeometry.RAMP_LANDING_FRACTION,
                                                  toFraction=dotGeometry.BLIND_TARGET_FRACTION,
                                                  cmd=cmd)
                # Diagnostic exposure post-blind-move (the move itself is
                # pfi-only / open-loop; this frame lets us inspect the
                # post-push detection state).
                self.cc.exposeAndExtractPositions()

            # Saving moves array
            np.save(dataPath / 'moves', moves)

            # INSTRM-2977: persist what the adaptive on-time loop learned, before the
            # next convergence resets it to 1.0 and the correction is lost.
            motorScales.saveMotorScales(self.cc, dataPath, cmd=cmd)

            # Blind-move inputs for moveToDotByFlux (in-memory, no file): it calls
            # blindMoveHiddenCobras once per flat, advancing fraction by dFraction.
            if len(dotGlobalIdx):
                self.dotScan = dict(dotGlobalIdx=np.asarray(dotGlobalIdx, dtype=int),
                                    commandedIdx=np.asarray(commandedIdx),
                                    moves=moves, dotGeom=dotGeom,
                                    # cobras sit at BLIND_TARGET_FRACTION after the blind move above
                                    fraction=dotGeometry.BLIND_TARGET_FRACTION,
                                    dFraction=0.05)   # per-flat scan increment

        except Exception:
            convergenceFailed = True
            raise

        finally:
            pfsConfig.convergenceParams['elapsedTime'] = round(time.time() - start, 3)
            self._finalizeWriteIngestPfsConfig(pfsConfig, cmd=cmd,
                                               convergenceFailed=convergenceFailed,
                                               notConvergedDistanceThreshold=notConvergedDistanceThreshold,
                                               validationMask=validationMask,
                                               cobraCommand=command)

        cmd.finish(f'text="We are at design position in {round(time.time() - start, 3)} seconds."')

    def _finalizeWriteIngestPfsConfig(self, pfsConfig, cmd,
                                      convergenceFailed=False, notConvergedDistanceThreshold=None,
                                      validationMask=None, cobraCommand=None):
        """Finalize pfsConfig, write to disk, ingest into opdb, and generate pfsConfig keyword."""
        atThetas = None if convergenceFailed else self.atThetas
        atPhis = None if convergenceFailed else self.atPhis
        cmd = cmd if cmd.isAlive() else self.actor.bcast
        try:
            maxIteration = pfsConfigUtils.finalize(pfsConfig,
                                                   finalIteration=self.actor.visitor.frameSeq - 1,
                                                   notConvergedDistanceThreshold=notConvergedDistanceThreshold,
                                                   validationMask=validationMask,
                                                   cobraCommand=cobraCommand,
                                                   atThetas=atThetas, atPhis=atPhis,
                                                   convergenceFailed=convergenceFailed)
        except Exception as e:
            if convergenceFailed:
                cmd.warn(f'text="pfsConfigUtils.finalize failed with: {e}"')
                return
            raise

        # So far so good.
        cmd.inform('text="pfsConfig updated successfully."')

        # write pfsConfig to disk.
        if not self.cc.simMode:
            pfsConfigUtils.writePfsConfig(pfsConfig, cmd=cmd)
        else:
            cmd.warn('text="in simulation mode so not writing pfsConfig file"')

        pfsConfigUtils.ingestPfsConfig(
            pfsConfig,
            allocated_at="now",
            cmd=cmd)

        cmd.inform(f'pfsConfig=0x{pfsConfig.pfsDesignId:016x},{pfsConfig.visit0},Done')

        return maxIteration

        cmd.finish('text="cobraMoveAngles completed"')

    def moveToDotByFlux(self, cmd):
        """Open-loop scan of dot cobras across the black dot for flat-field flux mapping.

        Called iteratively by iic, one SPS flat between each call.  moveToPfsDesign hands off
        ``self.dotScan`` (the inputs for dotGeometry.blindMoveHiddenCobras plus the running dot
        fraction).  Each call records the latest SPS flux, then calls blindMoveHiddenCobras to
        walk every *hidden* dot cobra one ``dFraction`` (0.05) deeper across the dot — bounded,
        motor-calibrated (fitPhiSpeed), phiFast=False (slow map), pure-FPGA moveSteps, no MCS.  Never stops
        or reverses; the flux-vs-depth minimum (dot centre) is analysed offline.

        Command keyword
        ---------------
        nRemaining : int (default 1)
            Remaining calls including this one.  Pass 0 for the final call: record flux,
            take no step, release the state.
        """
        from pfs.utils.database import opdb
        from ics.fpsActor.utils import dotGeometry

        cmdKeys = cmd.cmd.keywords
        nRemaining = cmdKeys['nRemaining'].values[0] if 'nRemaining' in cmdKeys else 1

        scan = self.dotScan
        if scan is None:
            cmd.fail('text="moveToDotByFlux: no dot-scan state - run moveToPfsDesign on a '
                     'BLACKSPOT (dot) design first"')
            return

        # -- record latest SPS flux (monitoring only; the profile is analysed offline) --
        try:
            db = opdb.OpDB()
            spsVisit = int(db.query_scalar('SELECT MAX(pfs_visit_id) FROM sps_visit'))
            fluxDf = db.query_dataframe(
                f'SELECT cobra_id, flux_norm FROM dot_roach_flux WHERE pfs_visit_id = {spsVisit}')
            nFlux = 0 if fluxDf.empty else int(np.isfinite(fluxDf.flux_norm.to_numpy()).sum())
            cmd.inform(f'text="dotScan flat: spsVisit={spsVisit} nFlux={nFlux}"')
        except Exception as e:
            cmd.warn(f'text="dotScan: flux read failed ({e}); continuing scan"')

        # -- walk the hidden dot cobras one dFraction deeper (skip the final record-only call) --
        if nRemaining > 0:
            f = scan['fraction']
            df = scan['dFraction']
            dotGeometry.blindMoveHiddenCobras(self.cc, scan['dotGlobalIdx'], scan['commandedIdx'],
                                              scan['moves'], scan['dotGeom'],
                                              fromFraction=f, toFraction=f + df, cmd=cmd)
            scan['fraction'] = f + df

        if nRemaining == 0:
            self.dotScan = None  # release for the next dotRoach sequence

        cmd.finish('text="moveToDotByFlux done"')

    def loadDotScales(self, cmd):
        """Load step scaling just for the dot traversal loop. """

        cmdKeys = cmd.cmd.keywords
        filename = cmdKeys['filename'].values[0] if 'filname' in cmdKeys else None

        cobras = self.cc.allCobras
        self.dotScales = np.ones(len(cobras))

        if filename is not None:
            scaling = pd.read_csv(filename)
            for i_i, phiScale in enumerate(scaling.itertuples()):
                cobraIdx = phiScale.cobra_id - 1
                self.dotScales[cobraIdx] = phiScale.scale

        cmd.finish(f'text="loaded {(self.dotScales != 0).sum()} phi scales"')

    def updateDotLoop(self, cmd):
        """ Move phi motors by a number of steps scaled by our internal dot scaling"""
        cmdKeys = cmd.cmd.keywords
        filename = cmdKeys['filename'].values[0]
        stepsPerMove = cmdKeys['stepsPerMove'].values[0]
        noMove = 'noMove' in cmdKeys

        cobras = self.cc.allCobras
        goodCobras = self.cc.goodIdx

        thetaSteps = np.zeros(len(cobras), dtype='i4')
        phiSteps = np.zeros(len(cobras), dtype='i4')

        moves = pd.read_csv(filename)
        allVisits = moves.visit.unique()
        lastVisit = np.sort(allVisits)[-1]
        cmd.inform(f'text="using dot mask for visit={lastVisit}, {len(goodCobras)} good cobras {goodCobras[:5]}"')
        for r_i, r in enumerate(moves[moves.visit == lastVisit].itertuples()):
            cobraIdx = r.cobraId - 1
            if r.keepMoving and cobraIdx in goodCobras:
                phiSteps[cobraIdx] = stepsPerMove * self.dotScales[cobraIdx]
                self.logger.info(f"{r_i} {r.cobraId} {phiSteps[cobraIdx]}")

        cmd.inform(f'text="moving={not noMove} {(phiSteps != 0).sum()} phi motors approx {stepsPerMove} steps')
        if not noMove:
            self.cc.pfi.moveSteps(cobras, thetaSteps, phiSteps, thetaFast=False, phiFast=False)

        cmd.finish(f'text="dot move done"')

    def testDotMove(self, cmd):
        """ Move phi motors by a number of steps scaled by our internal dot scaling"""
        cmdKeys = cmd.cmd.keywords
        stepsPerMove = cmdKeys['stepsPerMove'].values[0]

        cobras = self.cc.allCobras
        goodCobras = self.cc.goodCobras

        thetaSteps = np.zeros(len(cobras), dtype='i4')
        phiSteps = np.zeros(len(cobras), dtype='i4')

        for cobraIdx in range(len(cobras)):
            if cobraIdx + 1 not in goodCobras:
                phiSteps[cobraIdx] = stepsPerMove * self.dotScales[cobraIdx]
        self.logger.info("moving phi steps:", phiSteps)
        cmd.inform(f'text="moving {(phiSteps != 0).sum()} phi motors approx {stepsPerMove} steps')

        self.cc.pfi.moveSteps(cobras, thetaSteps, phiSteps, thetaFast=False, phiFast=False)
        self.testIteration(cmd, doFinish=False)
        cmd.finish(f'text="dot move done"')

    def calculateBoresight(self, cmd):

        """
        function for calculating the rotation centre
        """

        cmdKeys = cmd.cmd.keywords

        startFrame = cmdKeys['startFrame'].values[0]
        endFrame = cmdKeys['endFrame'].values[0]

        if 'writeToDB' in cmdKeys:
            writeToDB = True
        else:
            writeToDB = False

        cmd.inform(f'text="Write to DB is set to {writeToDB}."')

        # get a list of frameIds
        # two cases, for a single visitId, or multiple
        if (endFrame // 100 == startFrame // 100):
            frameIds = np.arange(startFrame, endFrame + 1)
        else:
            frameIds = np.arange(startFrame, endFrame + 100, 100)

        # use the pfsvisit id from the first frame for the database write
        pfsVisitId = startFrame // 100

        # the routine will calculate the value and write to db
        db = self.connectToDB(cmd)
        boresightMeasure.calcBoresight(cmd, db, frameIds, pfsVisitId, writeToDB = writeToDB)

        cmd.finish(f'text="Boresight calculation is finished."')

    def updateCobrasCenters(self, cmd):
        """Update calibModel.centers from the last moveToHome MCS frame.

        With @brokenOnly (default): updates only non-OK cobras, setting their center
        directly to the observed spot position (angles unknown).
        Without @brokenOnly: also updates good cobras by deriving the true center
        from the observed tip position using forward-kinematics geometry at home.

        Per-cobra info (old/new center, dx, dy, dist) is reported for non-OK cobras.
        For good cobras, percentile statistics (p1/p50/p99) on dx, dy, dist are reported.
        """
        brokenOnly = 'brokenOnly' in cmd.cmd.keywords

        label = 'non-OK' if brokenOnly else 'all'
        cmd.inform(f'text="Updating {label} cobra centers from last moveToHome frame..."')

        db = self.connectToDB(cmd)
        cobraIds, isGood, oldX, oldY, newX, newY = updateCobraCenters(self.cc.calibModel, db=db, brokenOnly=brokenOnly)

        dx = newX - oldX
        dy = newY - oldY
        dist = np.hypot(dx, dy)

        # Per-cobra report for non-OK cobras (small set, detailed output useful)
        brokenMask = ~isGood
        for cobraId, ox, oy, nx, ny, ddx, ddy, d in zip(
                cobraIds[brokenMask], oldX[brokenMask], oldY[brokenMask],
                newX[brokenMask], newY[brokenMask],
                dx[brokenMask], dy[brokenMask], dist[brokenMask]):
            cmd.inform(f'text="cobra {cobraId}: '
                       f'old=({ox:.3f},{oy:.3f}) new=({nx:.3f},{ny:.3f}) '
                       f'dx={ddx:+.3f} dy={ddy:+.3f} dist={d:.3f} mm"')

        # Percentile report for good cobras
        if isGood.any():
            for quantity, arr in [('dx', dx[isGood]), ('dy', dy[isGood]), ('dist', dist[isGood])]:
                p1, p50, p99 = np.percentile(arr, [1, 50, 99])
                cmd.inform(f'text="good cobras {quantity}: p1={p1:+.4f} p50={p50:+.4f} p99={p99:+.4f} mm"')

        cmd.finish(f'text="{len(cobraIds)} cobra centers updated '
                   f'({isGood.sum()} good, {brokenMask.sum()} non-OK)"')