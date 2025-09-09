import enum
import logging

import numpy as np
import pandas as pd
from ics.fpsActor.utils.alfUtils import sgfm, circleIntersections, robustRms, circleIntersectionsOrClosest
from ics.fpsActor.utils.kalmanTracker import KalmanAngleTracker3D
from scipy.interpolate import interp1d


class DotModel:
    x = np.array([0.00134, 0.00446, 0.02225, 0.15849, 0.26571, 0.39014,
                  0.53702, 0.56847, 0.65765, 0.76933, 0.86632, 0.92423, 1])
    y = np.array([0.21676, 0.29802, 0.37928, 0.46054, 0.54181, 0.62307,
                  0.70433, 0.78559, 0.86685, 0.94812, 1.02938, 1.11064, 1.3])

    model = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
    inverse_model = interp1d(y, x, kind='linear', bounds_error=False, fill_value=np.nan)

    @staticmethod
    def inferDistFromAttenuation(attenuations):
        """Estimate distance from attenuation (throughput)."""
        arr = np.atleast_1d(attenuations).astype(float)
        dist = DotModel.model(arr)
        dist[arr < DotModel.x[0]] = 0
        dist[arr >= DotModel.x[-1]] = DotModel.y[-1]
        return dist[0] if np.isscalar(attenuations) else dist

    @staticmethod
    def inferAttenuationFromDistance(distances):
        """Estimate attenuation (throughput) from distance."""
        arr = np.atleast_1d(distances).astype(float)
        att = DotModel.inverse_model(arr)
        att[arr < DotModel.y[0]] = 0
        att[arr > DotModel.y[-1]] = DotModel.x[-1]
        return att[0] if np.isscalar(distances) else att


class Flag(enum.IntFlag):
    BROKEN = 1 << 0
    STANDBY = 1 << 1
    MISBEHAVING = 1 << 2
    PHICENTER_UNKNOWN = 1 << 3
    NOT_CROSSING_DOT = 1 << 4
    BLOCKED = 1 << 5
    REVERSED = 1 << 6


class SingleRoach(object):
    ANGLE_TO_DOT = -0.3  # radians

    def __init__(self, driver, cobraId, thetaX, thetaY):
        self.driver = driver
        row = sgfm[sgfm.cobraId == cobraId].squeeze()

        self.scienceFiberId = row["scienceFiberId"]
        self.cobraId = row["cobraId"]
        self.fiberId = row["fiberId"]
        self.spectrographId = row["spectrographId"]
        self.COBRA_OK_MASK = row["COBRA_OK_MASK"]
        self.x = thetaX
        self.y = thetaY
        self.xDot = row["xDot"]
        self.yDot = row["yDot"]
        self.rDot = row["rDot"]
        self.armLength = row["armLength"]
        self.L1 = row["L1"]
        self.L2 = row["L2"]

        self.logger = driver.logger
        self.nearDotConvergenceDf = driver.convergenceDf[self.driver.convergenceDf.cobraId == self.cobraId].squeeze()
        self.fixedScalingDf = driver.fixedScalingDf[self.driver.fixedScalingDf.cobra_id == self.cobraId].sort_values(
            'iteration').reset_index(drop=True)

        self.phiCenterX, self.phiCenterY = np.nan, np.nan
        self.tracker = None
        self.dotEnterEdgeAngle = None
        self.dotExitEdgeAngle = None
        self.openLoopSteps = None
        self.targetAngle = None
        self.radialRms = np.nan
        self.initialVelocity = np.nan

        self.angleInputs = []
        self.predictions = []

        self.spotRows = []
        self.spotsArray = np.empty((0, 5))
        self.fiberThroughput = []

        self.statusFlag = 0

    @property
    def spotsDf(self):
        return pd.DataFrame(self.spotRows).reset_index(drop=True).sort_values('iteration')

    @property
    def mcsAngles(self):
        return self.spotsArray[:, 0]

    @property
    def statusStr(self):
        """Human-readable summary of active status flags."""
        return '|'.join([f.name for f in Flag if self.statusFlag & f.value])

    @property
    def stepScale(self):
        """Estimated steps per radian based on initial velocity."""
        return abs(self.driver.fixedSteps / self.initialVelocity)

    def calculateAngle(self, xMm, yMm):
        """Compute cobra angle (rad) from current position and phi center."""
        dx = xMm - self.phiCenterX
        dy = yMm - self.phiCenterY

        return np.arctan2(dy, dx)

    def calculateCoordinates(self, angle):
        """Compute (x, y) position for a given angle from phi center."""
        x = self.phiCenterX + self.L2 * np.cos(angle)
        y = self.phiCenterY + self.L2 * np.sin(angle)

        return x, y

    @property
    def doTrackCobra(self):
        """Return True if cobra is eligible for tracking updates."""
        return self.statusFlag in [0, 2]

    @property
    def doDriveCobra(self):
        """Return True if cobra is healthy and ready to be moved."""
        return self.statusFlag == 0

    def setStatusFlag(self, flag):
        """Set a status flag and log the updated status."""
        if self.statusFlag & flag:
            return

        self.statusFlag |= flag
        self.logger.info(f"[Cobra {self.cobraId}] SET flag: {Flag(flag).name} ({flag}). "
                         f"StatusStr: {self.statusStr}")

    def unsetStatusFlag(self, flag):
        """Clear a status flag and log the updated status."""
        if not self.statusFlag & flag:
            return

        self.statusFlag &= ~flag
        self.logger.info(f"[Cobra {self.cobraId}] UNSET flag: {Flag(flag).name} ({flag}). "
                         f"StatusStr: {self.statusStr}")

    def characterize(self, radialRmsThreshold, stepScaleThreshold):
        """Evaluate cobra health and flag issues based on geometry and motion."""
        if not self.COBRA_OK_MASK:
            self.setStatusFlag(Flag.BROKEN)
            return

        if any(np.isnan([self.phiCenterX, self.phiCenterY])):
            self.setStatusFlag(Flag.PHICENTER_UNKNOWN)
            return

        if np.isnan(self.radialRms) or self.radialRms > radialRmsThreshold:
            self.setStatusFlag(Flag.MISBEHAVING)

        if self.stepScale > stepScaleThreshold:
            self.setStatusFlag(Flag.BLOCKED)

        if self.initialVelocity > 0:
            self.setStatusFlag(Flag.REVERSED)

        interPhiDot = circleIntersections(self.phiCenterX, self.phiCenterY, self.L2, self.xDot, self.yDot, self.rDot)

        if not len(interPhiDot):
            self.setStatusFlag(Flag.NOT_CROSSING_DOT)

        else:
            dotEdgeAngles = np.array([self.calculateAngle(x, y) for x, y in interPhiDot])
            self.dotEnterEdgeAngle = dotEdgeAngles[np.argmin(abs(dotEdgeAngles - self.mcsAngles[-1]))]
            self.dotExitEdgeAngle = dotEdgeAngles[np.argmax(abs(dotEdgeAngles - self.mcsAngles[-1]))]

    def updatePhiCenter(self, scalingDf):
        """Estimate phi center from circle intersections over multiple iterations."""

        def fastFindPhiCenter(x, y, cobraTuple):
            """Compute optimal phi center as circle intersection closest to home-target angular difference."""
            xTheta, yTheta, L1, L2, xTarget, yTarget = cobraTuple
            intersections = circleIntersections(xTheta, yTheta, L1, x, y, L2)

            if not intersections:
                raise ValueError("No valid intersection points found between the circles.")

            angleDiffs = []
            for cx, cy in intersections:
                angleHome = np.arctan2(yTheta - cy, xTheta - cx)
                angleTarget = np.arctan2(yTarget - cy, xTarget - cx)
                angleDiff = (angleTarget - angleHome) % (2 * np.pi)
                angleDiffs.append(angleDiff)

            bestIndex = np.argmin(angleDiffs)
            return intersections[bestIndex]

        def robustFindPhiCenter(cobraData, scalingDf):
            """Estimate phi center as median of circle intersections across iterations."""
            cobraTuple = (cobraData.x, cobraData.y, cobraData.L1, cobraData.L2, cobraData.xTarget, cobraData.yTarget)
            xi = np.concatenate([[cobraData.xPosition], scalingDf.pfi_center_x_mm.to_numpy()])
            yi = np.concatenate([[cobraData.yPosition], scalingDf.pfi_center_y_mm.to_numpy()])

            robustPhi = []
            for x, y in zip(xi, yi):
                try:
                    robustPhi.append(fastFindPhiCenter(x, y, cobraTuple))
                except ValueError:
                    robustPhi.append((np.nan, np.nan))

            robustPhi = np.array(robustPhi)
            return np.nanmedian(robustPhi, axis=0)

        self.phiCenterX, self.phiCenterY = robustFindPhiCenter(self.nearDotConvergenceDf, scalingDf)

    def bootstrap(self):
        """Initialize phi center, spot history, radial RMS, and velocity f
            q_acceleration=self.driver.params[2],rom scaling data."""
        if not self.COBRA_OK_MASK:
            return

        self.updatePhiCenter(self.fixedScalingDf)

        for j, iterRow in self.fixedScalingDf.iterrows():
            self.addSpotInfo(iterRow)

        self.radialRms = self.calculateRadialRms()
        self.initialVelocity = self.calculateInitialVelocity()

    def setupTracker(self):
        """Initialize and warm up Kalman tracker using past MCS angle measurements."""
        if not self.doDriveCobra:
            return

        self.tracker = KalmanAngleTracker3D(initialAngle=self.mcsAngles[0],
                                            initialVelocity=self.initialVelocity,
                                            initialAcceleration=0,
                                            q_angle=self.driver.params[0],
                                            q_velocity=self.driver.params[1],
                                            q_acceleration=self.driver.params[2],
                                            r_measurement=self.driver.params[3])

        for i in range(len(self.mcsAngles) - 1):
            self.predictions.append(np.hstack((1, self.tracker.predict(steps=1))))
            self.updateTracker(self.mcsAngles[i + 1])

    def rebuildTracker(self):
        """Rebuild Kalman tracker with updated angles and re-feed all MCS measurements."""

        def weightFromThroughput(tp, t0=0.2, t1=0.6):
            """Compute measurement weight based on throughput using linear interpolation between t0 and t1."""
            return np.clip((tp - t0) / (t1 - t0), 0, 1)

        if not self.doDriveCobra:
            return

        steps = np.array(self.predictions)[:, 0].copy()

        self.angleInputs = [self.mcsAngles[0]]
        self.predictions.clear()

        self.initialVelocity = self.calculateInitialVelocity()

        self.tracker = KalmanAngleTracker3D(
            initialAngle=self.mcsAngles[0],
            initialVelocity=self.initialVelocity,
            initialAcceleration=0,
            q_angle=self.driver.params[0],
            q_velocity=self.driver.params[1],
            q_acceleration=self.driver.params[2],
            r_measurement=self.driver.params[3],
        )

        for i in range(len(steps)):
            predicted_angle, velocity, accel = self.tracker.predict(steps=steps[i])
            measured_angle, peak, fwhm, masked, mcsThroughput = self.spotsArray[i + 1]
            weight = weightFromThroughput(mcsThroughput)
            angleInput = weight * measured_angle + (1 - weight) * predicted_angle

            self.updateTracker(angleInput)
            self.predictions.append((steps[i], predicted_angle, velocity, accel))

    def addSpotInfo(self, iterRow):
        """Add angle and spot info from an MCS iteration row."""
        if iterRow.spot_id == -1 or self.statusFlag & Flag.PHICENTER_UNKNOWN:
            angle = np.nan
        else:
            angle = self.calculateAngle(iterRow.pfi_center_x_mm, iterRow.pfi_center_y_mm)

        self.spotsArray = np.append(self.spotsArray, np.array([[angle, iterRow.peakvalue, iterRow.fwhm, 0, np.nan]]),
                                    axis=0)
        self.spotRows.append(iterRow)

    def newMcsIteration(self, iterRow, nSigma=2.5):
        """Process a new MCS spot measurement and update tracking."""
        # add spot_info no matter what.
        self.addSpotInfo(iterRow)

        # no need to go further, still interested to see spots_info that why
        if not self.doTrackCobra:
            return

        _, medPeakValue, medFwhm, _, _ = np.median(self.spotsArray, axis=0)
        _, sigPeakValue, sigFhwm, _, _ = np.std(self.spotsArray, axis=0, ddof=1)

        peakSigma = (medPeakValue - iterRow.peakvalue) / sigPeakValue
        fwhmSigma = (medFwhm - iterRow.fwhm) / sigFhwm

        combinedSigma = 0.3 * peakSigma + 0.7 * fwhmSigma

        # if peak is partially or completely hidden
        if combinedSigma > nSigma or np.isnan(self.mcsAngles[-1]):
            self.spotsArray[-1, 3] = 1
            self.setStatusFlag(Flag.STANDBY)

        elif self.doDriveCobra:
            self.updateTracker(self.mcsAngles[-1])

    def updateTracker(self, angle):
        """Feed a new measured angle to the Kalman tracker."""
        # I guess it doesn't hurt to keep it for now.
        if np.isnan(angle):
            self.setStatusFlag(Flag.STANDBY)
            return

        # keeping track of what was actually being fed to the kalman.
        self.angleInputs.append(angle)
        self.tracker.update(angle)

    def projectAngle(self, angle):
        """Return (x, y) position for a given angle using phi center and arm length."""
        dx = self.L2 * np.cos(angle)
        dy = self.L2 * np.sin(angle)

        projected_x = self.phiCenterX + dx
        projected_y = self.phiCenterY + dy

        return projected_x, projected_y

    def calculateRadialRms(self):
        """Compute RMS of radial deviation from expected arm length during MCS iterations."""
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
        """Estimate initial angular velocity from early MCS angle measurements."""
        if not self.COBRA_OK_MASK:
            return

        return np.nanmean(np.diff(self.mcsAngles)[:3])

    def tuneSteps(self, remainingMcsIteration, remainingSpsIteration):
        """Compute number of motor steps to reach target angle using Kalman prediction."""

        def calculateSteps(anglePerIteration):
            """
            Calculate how many fixed motor steps are needed to achieve the desired angle change,
            based on predicted motion from the Kalman tracker.
            """
            predictedAngle = self.tracker.predict_external(steps=1)[0]
            anglePerKalmanStep = self.angleInputs[-1] - predictedAngle

            if anglePerKalmanStep == 0:
                return 0, 0  # Avoid division by zero

            # Ratio of desired angle change to Kalman-predicted step
            useKalmanStep = -anglePerIteration / anglePerKalmanStep
            realSteps = int(round(self.driver.fixedSteps * useKalmanStep))
            return useKalmanStep, realSteps

        # MCS overshoots slightly to improve dot edge detection
        angleOverShoot = SingleRoach.ANGLE_TO_DOT * self.driver.mcsOverShoot
        mcsObjective = self.dotEnterEdgeAngle + angleOverShoot

        # Angular distances to objectives
        mcsDistance = mcsObjective - self.angleInputs[-1]

        # Target angular movement per iteration
        anglePerMcsIteration = mcsDistance / remainingMcsIteration if remainingMcsIteration else 0

        if self.targetAngle:
            spsDistance = self.targetAngle - self.angleInputs[-1]
            anglePerSpsIteration = spsDistance / remainingSpsIteration if remainingSpsIteration else 0
        else:
            anglePerSpsIteration = 1.5 * np.mean(np.diff(self.angleInputs)[-3:])

        # Choose which phase is active
        if remainingMcsIteration:
            useKalmanStep, realSteps = calculateSteps(anglePerMcsIteration)
        else:
            useKalmanStep, realSteps = calculateSteps(anglePerSpsIteration)

        # Also store open-loop steps to target from current position
        _, openLoopSteps = calculateSteps(anglePerSpsIteration)
        self.openLoopSteps = openLoopSteps

        # Predict and store next angle from Kalman for tracking
        self.predictions.append(np.hstack((useKalmanStep, self.tracker.predict(steps=useKalmanStep))))

        return realSteps

    def getStepsToDot(self, remainingMcsIteration, remainingSpsIteration):
        """Return number of steps toward dot using Kalman-guided tuning."""
        if self.doDriveCobra:
            steps = self.tuneSteps(remainingMcsIteration, remainingSpsIteration)
        elif self.statusFlag & Flag.STANDBY or self.statusFlag & Flag.BROKEN:
            steps = 0
        else:
            steps = self.driver.fixedSteps  # just to get data

        return steps

    def getStepsOpenLoop(self):
        """Return number of open-loop steps to dot if Kalman is not usable."""
        if self.statusFlag & Flag.BROKEN:
            steps = 0
        elif self.openLoopSteps is not None:
            steps = self.openLoopSteps
        else:
            steps = self.driver.fixedSteps  # just to get data

        return steps

    def getTargetAngle(self, span=-np.pi / 3, nSteps=300):
        """Return angle along trajectory where estimated throughput is minimal."""
        angles = self.angleInputs[-1] + np.linspace(0, span, nSteps)
        xi, yi = self.calculateCoordinates(angles)
        distances = np.hypot(self.xDot - xi, self.yDot - yi)
        throughputs = DotModel.inferAttenuationFromDistance(distances)

        zeroMask = throughputs == 0
        if np.any(zeroMask):
            # Find contiguous sequences of zeros
            zeroIndices = np.where(zeroMask)[0]
            diffs = np.diff(zeroIndices)
            splitPoints = np.where(diffs > 1)[0] + 1
            groups = np.split(zeroIndices, splitPoints)

            # Take the largest zero block (or the first if tie)
            longestGroup = max(groups, key=len)
            centerIdx = longestGroup[len(longestGroup) // 2]
        else:
            centerIdx = np.nanargmin(throughputs)

        return angles[centerIdx]

    def getCobraSide(self):
        """Determine on which side of the dot the cobra currently is based on the throughput history."""
        n = len(self.fiberThroughput)
        if n <= 1:
            return 0
        elif n == 2:
            return np.sign(self.fiberThroughput[1] - self.fiberThroughput[0])

        side = -1
        prev_grad = np.sign(self.fiberThroughput[1] - self.fiberThroughput[0])

        for i in range(2, n):
            grad = np.sign(self.fiberThroughput[i] - self.fiberThroughput[i - 1])
            if grad > 0 and prev_grad < 0:
                side *= -1
            prev_grad = grad

        return side

    def concludeMcsPhase(self, A_opt=9.22590, p_opt=2.03226, q_opt=0.56524):
        """Finalize MCS tracking by updating phi center, refitting angles, and recalculating target angle."""

        def interpolateAngleAtHalfThroughput(dfEdgeDot):
            """Interpolate angle corresponding to 50% throughput (dot edge crossing)."""
            x = dfEdgeDot.mcsThroughput.to_numpy()
            y = dfEdgeDot.angles.to_numpy()

            isort = np.argsort(x)
            x = x[isort]
            y = y[isort]

            interp = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
            return float(interp(0.5))

        spotsDf = self.spotsDf.copy()
        spotsDf['estimatedFlux'] = A_opt * spotsDf['fwhm'] ** p_opt * spotsDf['peakvalue'] ** q_opt

        # Mask spots where the peak was flagged (e.g., hidden)
        spot_mask = ~self.spotsArray[:, 3].astype(bool)
        goodSpots = spotsDf[spot_mask].copy()
        goodSpots['iteration'] = spotsDf.iteration.astype('int32')

        # Update phi center using good spots only
        self.updatePhiCenter(goodSpots)

        # Recompute angles based on new phi center
        newAngles = np.array([self.calculateAngle(r.pfi_center_x_mm, r.pfi_center_y_mm) for _, r in spotsDf.iterrows()])
        self.spotsArray[:, 0] = newAngles

        spotsDf['mcsThroughput'] = spotsDf['estimatedFlux'] / np.median(goodSpots.estimatedFlux.to_numpy())
        spotsDf['mcsThroughput'] = spotsDf['mcsThroughput'].fillna(0)
        spotsDf['angles'] = newAngles

        self.spotsArray[:, 4] = spotsDf.mcsThroughput.to_numpy()

        # resetting standby flag.
        self.unsetStatusFlag(Flag.STANDBY)

        # Rebuild Kalman filter with corrected angles
        self.rebuildTracker()

        iMasked = np.argmin(spot_mask)

        if not iMasked:
            return

        dfEnterDot = spotsDf.loc[iMasked - 1:iMasked + 1].copy()
        dfEnterDot['mcsThroughput'] = dfEnterDot['mcsThroughput'] - dfEnterDot['mcsThroughput'].iloc[0] + 1

        self.dotEnterEdgeAngle = interpolateAngleAtHalfThroughput(dfEnterDot)

    def newSpsIteration(self, throughput, span=np.pi / 6, nSteps=100, maxDotUpdate=0.2):
        """Infer cobra angle from SPS throughput and update Kalman filter."""
        if not self.doTrackCobra or np.isnan(throughput):
            return

        self.fiberThroughput.append(throughput)
        side = self.getCobraSide()

        # Not enough points to determine side yet
        if side not in [-1, 1]:
            return

        # Estimate new dot center and update target angle if not already done
        if self.targetAngle is None:
            self.updateDotPosition(maxDotUpdate=maxDotUpdate)
            # Now that dot position estimate is correct, get updated target angle
            self.targetAngle = self.getTargetAngle()

        inferredAngle = self.inferAngleFromThroughput(throughput, side=side, span=span, nSteps=nSteps)
        self.updateTracker(inferredAngle)

    def updateDotPosition(self, maxDotUpdate):
        """Estimate new dot center based on SPS throughput and update target angle."""
        (xe, x1, x2), (ye, y1, y2) = self.calculateCoordinates([self.dotEnterEdgeAngle,
                                                                self.angleInputs[-1],
                                                                self.predictions[-1][1]])
        dist1 = DotModel.inferDistFromAttenuation(self.fiberThroughput[0])
        dist2 = DotModel.inferDistFromAttenuation(self.fiberThroughput[1])

        intersections = circleIntersectionsOrClosest(x1, y1, dist1, x2, y2, dist2)
        if not intersections:
            return

        # Evaluate intersection points against dot edge
        theta = np.linspace(0, 2 * np.pi, 100)
        xci = np.cos(theta) * self.rDot + xe
        yci = np.sin(theta) * self.rDot + ye

        valid = []
        for xDotCand, yDotCand in intersections:
            distToEdge = np.hypot(xci - xDotCand, yci - yDotCand)
            iMin = np.argmin(distToEdge)
            distToPrevDot = np.hypot(xci[iMin] - self.xDot, yci[iMin] - self.yDot)

            if distToPrevDot <= maxDotUpdate:
                valid.append((distToEdge[iMin], iMin))

        if valid:
            _, iBest = min(valid)
            self.xDotNew = xci[iBest]
            self.yDotNew = yci[iBest]

    def inferAngleFromThroughput(self, throughput, side, span, nSteps):
        """Infer cobra angular position from SPS throughput estimate."""
        dist = DotModel.inferDistFromAttenuation(throughput)
        if not dist:
            return self.targetAngle

        if side == -1:
            angles = self.targetAngle + np.linspace(0, span, nSteps)
        else:
            angles = self.targetAngle - np.linspace(0, span, nSteps)

        xi, yi = self.calculateCoordinates(angles)
        distances = np.hypot(self.xDot - xi, self.yDot - yi)

        bestIdx = np.nanargmin(np.abs(distances - dist))
        return angles[bestIdx]


class HotRoachDriver(object):
    """Driver managing all SingleRoach instances and iteration coordination for Cobra tracking."""

    params = [1.000e-01, 9.996e-02, 5.974e-02, 1.294e-02]

    def __init__(self, convergenceDf, fixedScalingDf, fixedSteps, mcsOverShoot=0.2):
        """Initialize the HotRoachDriver with convergence data and global parameters."""
        self.convergenceDf = convergenceDf
        self.fixedScalingDf = fixedScalingDf
        self.fixedSteps = fixedSteps
        self.mcsOverShoot = mcsOverShoot
        self.logger = logging.getLogger('HotRoach')

        self.roaches = dict()

    def bootstrap(self, thetaX, thetaY):
        """Bootstrap all SingleRoach objects and initialize their tracking states."""
        for cobraId, cobraData in self.convergenceDf.groupby('cobraId'):
            self.roaches[cobraId] = SingleRoach(self, cobraId, thetaX[cobraId - 1], thetaY[cobraId - 1])
            self.roaches[cobraId].bootstrap()

        radialRmsThreshold = self.calculateRmsThreshold()
        stepScaleThreshold = self.calculateStepScaleThreshold()

        for cobraId, roach in self.roaches.items():
            roach.characterize(radialRmsThreshold=radialRmsThreshold, stepScaleThreshold=stepScaleThreshold)
            roach.setupTracker()

    def calculateRmsThreshold(self, nSigma=10):
        """Calculate radial RMS threshold for outlier detection using robust RMS."""
        radialRms = np.array([roach.radialRms for roach in self.roaches.values()])
        rms = robustRms(radialRms)
        threshold = np.nanmedian(radialRms) + nSigma * rms
        return threshold

    def calculateStepScaleThreshold(self, nSigma=50):
        """Calculate step scaling threshold for outlier detection using robust RMS."""
        stepScales = np.array([roach.stepScale for roach in self.roaches.values()])
        rms = robustRms(stepScales)
        threshold = np.nanmedian(stepScales) + nSigma * rms
        return threshold

    def makeScalingDf(self, remainingMcsIteration, remainingSpsIteration, doOpenLoop=False):
        """Create a DataFrame of Cobra movement steps for the next iteration."""
        res = []

        for cobraId, roach in self.roaches.items():
            if doOpenLoop:
                steps = roach.getStepsOpenLoop()
            else:
                steps = roach.getStepsToDot(remainingMcsIteration, remainingSpsIteration)

            bitMask = int(steps != 0)

            res.append((cobraId, bitMask, steps))

        return pd.DataFrame(res, columns=['cobraId', 'bitMask', 'steps'])

    def newMcsIteration(self, cobraMatch):
        """Update each SingleRoach with a new MCS iteration measurement."""
        for cobraId, roach in self.roaches.items():
            roach.newMcsIteration(cobraMatch[cobraMatch.cobra_id == roach.cobraId].squeeze())

    def concludeMcsPhase(self):
        """Finalize the MCS phase for all active cobras."""
        for cobraId, roach in self.roaches.items():
            if not roach.doTrackCobra:
                continue

            roach.concludeMcsPhase()

    def newSpsIteration(self, fluxDf):
        """Process SPS throughput measurements for each cobra."""
        for cobraId, roach in self.roaches.items():
            if not roach.doTrackCobra:
                continue

            fluxNorm = fluxDf[fluxDf.cobraId == cobraId].sort_values('nIter').fluxNorm.to_numpy()
            throughput = fluxNorm[-1] / fluxNorm[0]
            roach.newSpsIteration(throughput)
