import enum

import numpy as np
import pandas as pd
from ics.fpsActor.utils.alfUtils import sgfm, circleIntersections, findPhiCenter, robustRms
from ics.fpsActor.utils.kalmanTracker import KalmanAngleTracker3D
from scipy.interpolate import interp1d


class DotModel:
    x = [0.00132981, 0.00230789, 0.01474092, 0.18052107, 0.38697369,
         0.55950376, 0.70959454, 0.89372059, 0.99174909, 1.0]
    y = [0.0, 0.07896623, 0.14536758, 0.20218487, 0.25713816,
         0.31589083, 0.37996429, 0.43791802, 0.49316161, 0.55856142]

    model = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)

    @staticmethod
    def inferDistFromAttenuation(attenuations):
        # Convert to array if it's a scalar
        arr = np.atleast_1d(attenuations).astype(float)

        # Default result using interpolation
        dist = DotModel.model(arr)

        # Handle edge cases
        dist[arr < DotModel.x[0]] = 0
        dist[arr > 0.95] = np.nan

        # Return scalar if input was scalar
        return dist[0] if np.isscalar(attenuations) else dist


class Flag(enum.IntFlag):
    BROKEN = 1 << 0
    HIDDEN = 1 << 1
    MISBEHAVING = 1 << 2
    PHI_UNKNOWN = 1 << 3
    NOT_CROSSING_DOT = 1 << 4
    BLOCKED = 1 << 5
    REVERSED = 1 << 6


class SingleRoach(object):

    def __init__(self, driver, cobraId):
        self.driver = driver
        row = sgfm[sgfm.cobraId == cobraId].squeeze()

        self.scienceFiberId = row["scienceFiberId"]
        self.cobraId = row["cobraId"]
        self.fiberId = row["fiberId"]
        self.spectrographId = row["spectrographId"]
        self.COBRA_OK_MASK = row["COBRA_OK_MASK"]
        self.x = row["x"]
        self.y = row["y"]
        self.xDot = row["xDot"]
        self.yDot = row["yDot"]
        self.rDot = row["rDot"]
        self.armLength = row["armLength"]
        self.L1 = row["L1"]
        self.L2 = row["L2"]

        self.phiCenterX, self.phiCenterY = None, None
        self.tracker = None
        self.dotEnterEdgeAngle = None
        self.dotExitEdgeAngle = None
        self.openLoopSteps = None
        self.radialRms = np.nan
        self.initialVelocity = np.nan

        self.angles = []
        self.predicted = []

        self.statusFlag = 0

    @property
    def nearDotConvergenceDf(self):
        return self.driver.convergenceDf[self.driver.convergenceDf.cobraId == self.cobraId].squeeze()

    @property
    def fixedScalingDf(self):
        return self.driver.fixedScalingDf[self.driver.fixedScalingDf.cobra_id == self.cobraId].sort_values(
            'iteration').reset_index(drop=True)

    @property
    def statusStr(self):
        return '|'.join([f.name for f in Flag if self.statusFlag & f.value])

    @property
    def dotCenterAngle(self):
        return self.calculateAngle(self.xDot, self.yDot)

    @property
    def stepScale(self):
        return abs(self.driver.fixedSteps / self.initialVelocity)

    def calculateAngle(self, xMm, yMm):
        dx = xMm - self.phiCenterX
        dy = yMm - self.phiCenterY

        return np.arctan2(dy, dx)

    @property
    def doTrackCobra(self):
        return self.statusFlag in [0]

    def setStatusFlag(self, radialRmsThreshold, stepScaleThreshold):
        if not self.COBRA_OK_MASK:
            self.statusFlag |= Flag.BROKEN
            return

        if any(np.isnan([self.phiCenterX, self.phiCenterY])):
            self.statusFlag |= Flag.PHI_UNKNOWN
            return

        if np.isnan(self.radialRms) or self.radialRms > radialRmsThreshold:
            self.statusFlag |= Flag.MISBEHAVING

        if self.stepScale > stepScaleThreshold:
            self.statusFlag |= Flag.BLOCKED

        if self.initialVelocity > 0:
            self.statusFlag |= Flag.REVERSED

        interPhiDot = circleIntersections(self.phiCenterX, self.phiCenterY, self.L2, self.xDot, self.yDot, self.rDot)

        if not len(interPhiDot):
            self.statusFlag |= Flag.NOT_CROSSING_DOT

        else:
            dotEdgeAngles = np.array([self.calculateAngle(x, y) for x, y in interPhiDot])
            self.dotEnterEdgeAngle = dotEdgeAngles[np.argmin(abs(dotEdgeAngles - self.angles[-1]))]
            self.dotExitEdgeAngle = dotEdgeAngles[np.argmax(abs(dotEdgeAngles - self.angles[-1]))]

    def updatePhiCenter(self, scalingDf):
        def robustFindPhiCenter(cobraData, scalingDf):
            robustPhi = []

            for iteration in range(-1, scalingDf.iteration.max() + 1):
                cobraData = cobraData.copy()

                if iteration != -1:
                    iterVal = scalingDf[scalingDf.iteration == iteration].squeeze()
                    cobraData['xPosition'] = iterVal.pfi_center_x_mm
                    cobraData['yPosition'] = iterVal.pfi_center_y_mm

                    try:
                        robustPhi.append(findPhiCenter(cobraData))
                    except:
                        robustPhi.append((np.nan, np.nan))

            robustPhi = np.array(robustPhi)
            return np.nanmedian(robustPhi, axis=0)

        self.phiCenterX, self.phiCenterY = robustFindPhiCenter(self.nearDotConvergenceDf, scalingDf)

    def bootstrap(self):
        if not self.COBRA_OK_MASK:
            return

        self.updatePhiCenter(self.fixedScalingDf)

        for j, iterRow in self.fixedScalingDf.iterrows():
            self.addAngle(iterRow)

        self.radialRms = self.calculateRadialRms()
        self.initialVelocity = self.calculateInitialVelocity()

    def setupTracker(self):
        if not self.doTrackCobra:
            return

        self.tracker = KalmanAngleTracker3D(initialAngle=self.angles[0],
                                            initialVelocity=self.initialVelocity,
                                            initialAcceleration=0,
                                            q_angle=self.driver.params[0],
                                            q_velocity=self.driver.params[1],
                                            q_acceleration=self.driver.params[2],
                                            r_measurement=self.driver.params[3])

        for i in range(len(self.angles) - 1):
            self.predicted.append(self.tracker.predict(steps=1))
            self.updateTracker(self.angles[i + 1])

    def addAngle(self, iterRow):
        if iterRow.spot_id == -1:
            angle = np.nan
        else:
            angle = self.calculateAngle(iterRow.pfi_center_x_mm, iterRow.pfi_center_y_mm)

        self.angles.append(angle)

    def addMcsIteration(self, iterRow, doUpdateTracker):
        self.addAngle(iterRow)

        if doUpdateTracker:
            self.updateTracker(self.angles[-1])

    def addSpsIteration(self, attenuation, mergeAngle):
        distanceToCenterDot = DotModel.inferDistFromAttenuation(attenuation)
        angle = self.dotCenterAngle + distanceToCenterDot

        if mergeAngle:
            self.angles[-1] = np.nanmean([self.angles[-1], angle])
        else:
            self.angles.append(angle)

        self.updateTracker(self.angles[-1])

    def updateTracker(self, angle):
        if np.isnan(angle):
            self.statusFlag |= Flag.HIDDEN
            return

        self.tracker.update(angle)

    def projectAngle(self, angle):
        dx = self.L2 * np.cos(angle)
        dy = self.L2 * np.sin(angle)

        projected_x = self.phiCenterX + dx
        projected_y = self.phiCenterY + dy

        return projected_x, projected_y

    def calculateRadialRms(self):
        df2 = self.fixedScalingDf[self.fixedScalingDf.spot_id != -1]

        if not len(df2):
            return np.nan

        x_mm = np.concatenate(([self.nearDotConvergenceDf.xPosition], df2.pfi_center_x_mm))
        y_mm = np.concatenate(([self.nearDotConvergenceDf.yPosition], df2.pfi_center_y_mm))

        # Compute deltas from center
        dx = x_mm - self.phiCenterX
        dy = y_mm - self.phiCenterY

        # Compute actual distances from center
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # Compute difference from expected radius
        radius_error = distance - self.L2

        # Compute a global indicator: sum of squared deviations
        radial_rms = np.sqrt(np.mean(radius_error ** 2))

        return radial_rms

    def calculateInitialVelocity(self):
        if not self.COBRA_OK_MASK:
            return

        return np.nanmean(np.diff(self.angles))

    def tuneSteps(self, remainingMcsIteration, remainingSpsIteration):
        def calculateSteps(anglePerIteration):
            predicted = self.tracker.predict_external(steps=1)[0]
            anglePerKalmanStep = self.angles[-1] - predicted
            useKalmanStep = anglePerIteration / anglePerKalmanStep
            useKalmanStep *= -1
            realSteps = int(round(self.driver.fixedSteps * useKalmanStep))
            return useKalmanStep, realSteps

        distanceCenterDot = self.dotCenterAngle - self.angles[-1]
        distanceEnterDot = self.dotEnterEdgeAngle - self.angles[-1]
        distanceExitDot = self.dotExitEdgeAngle - self.angles[-1]

        anglePerMcsIteration = distanceEnterDot / remainingMcsIteration
        anglePerSpsIteration = distanceCenterDot / remainingSpsIteration

        # anglePerIteration = np.mean([distanceEnterDot, distanceCenterDot]) / remainingIteration

        if remainingMcsIteration:
            useKalmanStep, realSteps = calculateSteps(anglePerMcsIteration)
        else:
            useKalmanStep, realSteps = calculateSteps(anglePerSpsIteration)

        _, openLoopSteps = calculateSteps(anglePerSpsIteration)

        self.openLoopSteps = openLoopSteps

        self.predicted.append(self.tracker.predict(steps=useKalmanStep))

        return realSteps

    def getStepsToDot(self, remainingMcsIteration, remainingSpsIteration):
        if self.doTrackCobra:
            steps = self.tuneSteps(remainingMcsIteration, remainingSpsIteration)
        elif self.statusFlag & Flag.HIDDEN or self.statusFlag & Flag.BROKEN:
            steps = 0
        else:
            steps = -60  # just to get data

        return steps

    def getStepsOpenLoop(self):
        if self.statusFlag & Flag.BROKEN:
            steps = 0
        elif self.openLoopSteps is not None:
            steps = self.openLoopSteps
        else:
            steps = -60  # just to get data

        return steps


class HotRoachDriver(object):
    params = [1.000e-01, 9.996e-02, 5.974e-02, 1.294e-02]

    def __init__(self, convergenceDf, fixedScalingDf, fixedSteps):
        self.convergenceDf = convergenceDf
        self.fixedScalingDf = fixedScalingDf
        self.fixedSteps = fixedSteps

        self.roaches = dict()

    def bootstrap(self):
        for cobraId, cobraData in self.convergenceDf.groupby('cobraId'):
            self.roaches[cobraId] = SingleRoach(self, cobraId)
            self.roaches[cobraId].bootstrap()

        radialRmsThreshold = self.calculateRmsThreshold()
        stepScaleThreshold = self.calculateStepScaleThreshold()

        for cobraId, roach in self.roaches.items():
            roach.setStatusFlag(radialRmsThreshold=radialRmsThreshold, stepScaleThreshold=stepScaleThreshold)
            roach.setupTracker()

    def calculateRmsThreshold(self, nSigma=10):
        radialRms = np.array([roach.radialRms for roach in self.roaches.values()])
        rms = robustRms(radialRms)
        threshold = np.nanmedian(radialRms) + nSigma * rms
        return threshold

    def calculateStepScaleThreshold(self, nSigma=50):
        stepScales = np.array([roach.stepScale for roach in self.roaches.values()])
        rms = robustRms(stepScales)
        threshold = np.nanmedian(stepScales) + nSigma * rms
        return threshold

    def makeScalingDf(self, remainingMcsIteration, remainingSpsIteration, doOpenLoop=False):
        res = []

        for cobraId, roach in self.roaches.items():
            if doOpenLoop:
                steps = roach.getStepsOpenLoop()
            else:
                steps = roach.getStepsToDot(remainingMcsIteration, remainingSpsIteration)

            bitMask = int(steps != 0)

            res.append((cobraId, bitMask, steps))

        return pd.DataFrame(res, columns=['cobraId', 'bitMask', 'steps'])

    def newMcsIteration(self, cobraMatch, doUpdateTracker):
        for cobraId, roach in self.roaches.items():
            if not roach.doTrackCobra:
                continue

            roach.addMcsIteration(cobraMatch[cobraMatch.cobra_id == roach.cobraId].squeeze(),
                                  doUpdateTracker=doUpdateTracker)

    def newSpsIteration(self, fluxDf, mergeAngle):
        for cobraId, roach in self.roaches.items():
            if roach.statusFlag & Flag.HIDDEN:
                roach.statusFlag &= ~Flag.HIDDEN

            if not roach.doTrackCobra:
                continue

            fluxNorm = fluxDf[fluxDf.cobraId == cobraId].sort_values('nIter').fluxNorm.to_numpy()
            attenuation = fluxNorm[-1] / fluxNorm[0]

            roach.addSpsIteration(attenuation, mergeAngle=mergeAngle)
