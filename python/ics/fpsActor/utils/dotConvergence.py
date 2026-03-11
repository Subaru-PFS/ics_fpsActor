import numpy as np
from ics.fpsActor.utils.alfUtils import sgfm, circleIntersections
from pfs.utils.database import opdb


class DotConverger:
    """Algorithm logic for moving cobras toward black dot positions.

    Holds references to cobraCoach state (atThetas, atPhis) and mutates
    them in place so changes are reflected back in the caller.

    Parameters
    ----------
    cc : cobraCoach
        The cobra coach instance.
    atThetas : `numpy.ndarray`
        Current theta angles for all cobras (radians). Mutated in place.
    atPhis : `numpy.ndarray`
        Current phi angles for all cobras (radians). Mutated in place.
    """

    def __init__(self, cc, atThetas, atPhis):
        self.cc = cc
        self.atThetas = atThetas
        self.atPhis = atPhis
        self.db = opdb.OpDB()

    def goodIdxWhere(self, mask):
        """Return good cobra indices where mask is True."""
        if len(mask) != len(self.cc.allCobras):
            raise ValueError(f'mask size {len(mask)} does not match allCobras size {len(self.cc.allCobras)}')
        return np.intersect1d(self.cc.goodIdx, np.where(mask)[0])

    def getDotPosition(self):
        return sgfm.xDot.to_numpy() + 1j * sgfm.yDot.to_numpy()

    def fetchLastMeasuredPositions(self):
        """Fetch cobra measured positions (mm) from the DB for the last MCS frame.

        Queries max(mcs_frame_id) to identify the most recent frame, then retrieves
        the corresponding cobra_match rows.

        Returns
        -------
        `pandas.DataFrame`
            DataFrame ordered by cobra_id with columns including
            pfi_center_x_mm, pfi_center_y_mm and spot_id (-1 if undetected).
        """
        lastFrameNum = self.db.query_dataframe('select max(mcs_frame_id) from mcs_data').squeeze()
        pfs_visit_id = lastFrameNum // 100
        iteration = lastFrameNum % 100

        sql = (
            'SELECT cm.pfs_visit_id, cm.iteration, cm.cobra_id, cm.spot_id, cm.pfi_center_x_mm, cm.pfi_center_y_mm '
            'FROM cobra_match cm LEFT OUTER JOIN mcs_data m '
            'ON m.spot_id = cm.spot_id AND m.mcs_frame_id = cm.mcs_frame_id '
            f'WHERE cm.pfs_visit_id = {pfs_visit_id} AND cm.iteration = {iteration} '
            'ORDER BY cm.cobra_id ASC')

        return self.db.query_dataframe(sql)

    def getDetectedIdx(self, measured=None):
        """Return good cobra indices detected in the last MCS frame."""
        if measured is None:
            measured = self.fetchLastMeasuredPositions()

        detected = measured.spot_id.to_numpy() != -1
        return self.goodIdxWhere(detected)

    def computeDotEdgeTargets(self):
        """Compute target angles at the nearest dot edge for all cobras.

        For each cobra, finds the intersection of the phi arm circle (centered
        at the elbow, radius L2) with the dot circle. circleIntersections returns
        the two solutions with the nearest phi (closest to home) first.

        Returns
        -------
        thetaTarget, phiTarget : `numpy.ndarray`
            Target theta and phi angles (radians) for each cobra.
        """
        # Get cobra angles when pointing directly at the dot center.
        dotPos = self.getDotPosition()
        thetasToDot, _, _ = self.cc.pfi.positionsToAngles(self.cc.allCobras, dotPos)
        thetasToDot = thetasToDot[:, 0]

        # Find the elbow positions at those angles.
        elbows = self.cc.pfi.anglesToElbowPositions(self.cc.allCobras, thetasToDot)

        # Intersect phi arm circle with dot circle; take the nearest solution (smallest phi).
        nearDotEdge = []
        for i in range(len(elbows)):
            intersections = np.array(circleIntersections(np.real(elbows[i]), np.imag(elbows[i]), sgfm.iloc[i].L2,
                                                         sgfm.iloc[i].xDot, sgfm.iloc[i].yDot,
                                                         sgfm.iloc[i].rDot))
            nearDotEdge.append(intersections[0, 0] + 1j * intersections[0, 1])

        thetaTarget, phiTarget, _ = self.cc.pfi.positionsToAngles(self.cc.allCobras, nearDotEdge)
        return thetaTarget[:, 0], phiTarget[:, 0]

    def computeDotCenterTargets(self):
        """Compute target angles pointing directly at the dot center for all cobras.

        Returns
        -------
        thetaTarget, phiTarget : `numpy.ndarray`
            Target theta and phi angles (radians) for each cobra.
        """
        dotPos = self.getDotPosition()
        thetaTarget, phiTarget, _ = self.cc.pfi.positionsToAngles(self.cc.allCobras, dotPos)
        return thetaTarget[:, 0], phiTarget[:, 0]

    def retractOvershotCobras(self, cmd, phiTarget):
        """Retract cobras that have overshot past phiTarget to phi home.

        Cobras with phi > phiTarget are already past the dot edge and must be
        retracted to phi=0 before the convergence loop can approach from below.
        """
        overshotIdx = self.goodIdxWhere(self.atPhis > phiTarget)
        overshotCobras = self.cc.allCobras[overshotIdx]

        cmd.inform(f'text="Retracting {len(overshotIdx)} overshot cobras to phi home"')
        self.cc.moveToHome(overshotCobras, thetaEnable=False, phiEnable=True, thetaCCW=False, noMCS=False)

        self.cc.setCurrentAngles(overshotCobras, thetaAngles=None, phiAngles=np.zeros(len(overshotIdx)))
        self.atPhis[overshotIdx] = 0

    def convergeToDotTarget(self, cmd, thetaTarget, phiTarget, nIteration=12):
        """Closed-loop iterative convergence toward (thetaTarget, phiTarget).

        Each iteration moves by (remaining delta / remaining iterations), capped
        to avoid large corrections when phi is small and angle estimates are imprecise.
        activeIdx is refreshed each iteration from the current MCS frame.
        """
        maxDeltaTheta = np.deg2rad(5)  # tight cap: theta estimates are noisy when phi is small
        maxDeltaPhi = np.deg2rad(10)

        # Start with good cobras currently detected in MCS.
        activeIdx = self.getDetectedIdx()

        cmd.inform(f'text="Converging {len(activeIdx)} cobras to dot target in {nIteration} iterations"')

        for i in range(nIteration):
            remaining = nIteration - i

            # Compute per-step delta, spread evenly over remaining iterations.
            deltaThetas = np.clip((thetaTarget - self.atThetas) / remaining, -maxDeltaTheta, maxDeltaTheta)
            deltaPhis = np.clip((phiTarget - self.atPhis) / remaining, -maxDeltaPhi, maxDeltaPhi)

            activeCobras = self.cc.allCobras[activeIdx]
            cmd.inform(f'text="iteration {i + 1}/{nIteration} -- {len(activeIdx)} cobras active"')
            self.cc.moveDeltaAngles(activeCobras, deltaThetas[activeIdx], deltaPhis[activeIdx],
                                    thetaFast=False, phiFast=False)

            # Fetch measured positions and update angles only for detected cobras.
            measured = self.fetchLastMeasuredPositions()
            detected = (measured.spot_id != -1).to_numpy()

            measuredPos = measured.pfi_center_x_mm.to_numpy() + 1j * measured.pfi_center_y_mm.to_numpy()
            measuredThetas, measuredPhis, _ = self.cc.pfi.positionsToAngles(self.cc.allCobras, measuredPos)

            self.atThetas[detected] = measuredThetas[detected, 0]
            self.atPhis[detected] = measuredPhis[detected, 0]

            # Refresh active cobras from current MCS frame.
            activeIdx = self.getDetectedIdx(measured)
