import os

import numpy as np
import pandas as pd
from ics.fpsActor.utils.alfUtils import sgfm
from pfs.utils.database import opdb
from scipy.interpolate import interp1d

ROOT_DIR = '/data/fps'


class DotModel:
    """Empirical mapping from fiber attenuation to distance from dot center.

    The model was derived from geometry: x is the attenuation_norm (0 = fully
    obscured at dot center, 1 = full flux outside the dot) and y is the
    corresponding distance from the dot center in mm.

    Cobras with attenuation_norm > 0.95 are considered not yet under the dot
    and return NaN (no valid position estimate).
    """
    x = [0.00132981, 0.00230789, 0.01474092, 0.18052107, 0.38697369,
         0.55950376, 0.70959454, 0.89372059, 0.99174909, 1.0]
    y = [0.0, 0.07896623, 0.14536758, 0.20218487, 0.25713816,
         0.31589083, 0.37996429, 0.43791802, 0.49316161, 0.55856142]

    _model = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)

    @staticmethod
    def inferDistFromAttenuation(attenuations):
        """Estimate distance from dot center (mm) given attenuation_norm values.

        Parameters
        ----------
        attenuations : scalar or array-like
            attenuation_norm values (lamp- and reference-corrected flux ratio).

        Returns
        -------
        dist : scalar or `numpy.ndarray`
            Distance from dot center in mm. NaN for cobras not yet under the dot
            (attenuation > 0.95) or outside the interpolation range.
        """
        arr = np.atleast_1d(attenuations).astype(float)
        dist = DotModel._model(arr)
        dist[arr < DotModel.x[0]] = 0       # fully obscured → at center
        dist[arr > 0.95] = np.nan            # not yet under the dot
        return dist[0] if np.isscalar(attenuations) else dist


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

    BROKEN_COBRA = 1  # cobra not in goodIdx
    RETRACT_PHI = 2  # phi was above phiLimit at convergence start, needs phase 1 retraction
    NON_CROSSING_DOT = 4  # phi arm cannot geometrically reach the dot
    HIDDEN_WITH_MCS = 8  # cobra confirmed hidden under the dot
    FAILED_TO_HIDE = 16  # not hidden after all iterations

    def __init__(self, cc, atThetas, atPhis):
        self.cc = cc
        self.atThetas = atThetas
        self.atPhis = atPhis
        self.db = opdb.OpDB()

        self.allPhiSteps = [np.zeros(len(cc.allCobras))]
        self.allThetaSteps = [np.zeros(len(cc.allCobras))]
        self.allPhis = [atPhis.copy()]
        self.allThetas = [atThetas.copy()]
        self.allMcsFrameIds = []
        self.allSpsVisits = []
        self.allFlux = []
        self.allFluxNorm = []
        self.allFluxRatio = []
        self.allAttenuationNorm = []
        self.stepsToCenter = None
        self.visitId = None
        self.dumpVersion = 0

        nCobras = len(cc.allCobras)
        self.cobraFlags = np.zeros(nCobras, dtype=int)
        self.slewStartIter = np.zeros(nCobras, dtype=int)

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
        lastFrameNum = int(self.db.query_dataframe('select max(mcs_frame_id) from mcs_data').squeeze())
        pfs_visit_id = lastFrameNum // 100
        iteration = lastFrameNum % 100
        self.visitId = pfs_visit_id
        self.allMcsFrameIds.append(lastFrameNum)
        self.allSpsVisits.append(-1)
        self.allFlux.append(np.full(len(self.cc.allCobras), np.nan))
        self.allFluxNorm.append(np.full(len(self.cc.allCobras), np.nan))
        self.allFluxRatio.append(np.full(len(self.cc.allCobras), np.nan))
        self.allAttenuationNorm.append(np.full(len(self.cc.allCobras), np.nan))

        sql = (
            'SELECT cm.pfs_visit_id, cm.iteration, cm.cobra_id, cm.spot_id, cm.pfi_center_x_mm, cm.pfi_center_y_mm '
            'FROM cobra_match cm LEFT OUTER JOIN mcs_data m '
            'ON m.spot_id = cm.spot_id AND m.mcs_frame_id = cm.mcs_frame_id '
            f'WHERE cm.pfs_visit_id = {pfs_visit_id} AND cm.iteration = {iteration} '
            'ORDER BY cm.cobra_id ASC')

        return self.db.query_dataframe(sql)

    @classmethod
    def fromCsv(cls, filepath, cc):
        """Reconstruct a DotConverger from a CSV previously written by dumpConvergenceData.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        cc : cobraCoach
            cobraCoach instance (needed for geometry and goodIdx).

        Returns
        -------
        DotConverger
            Fully restored instance with stepsToCenter already computed.
        """
        df = pd.read_csv(filepath)
        nCobras = len(cc.allCobras)

        def pivot(col):
            return df.pivot(index='iteration', columns='cobraId', values=col).to_numpy()

        phiSteps_mat = pivot('phiSteps')
        thetaSteps_mat = pivot('thetaSteps')
        phis_mat = pivot('phi')
        thetas_mat = pivot('theta')

        # Last known position per cobra (last non-nan detected value).
        atThetas = np.zeros(nCobras)
        atPhis = np.zeros(nCobras)
        for i in range(nCobras):
            valid = np.isfinite(thetas_mat[:, i])
            if valid.any():
                atThetas[i] = thetas_mat[valid, i][-1]
            valid = np.isfinite(phis_mat[:, i])
            if valid.any():
                atPhis[i] = phis_mat[valid, i][-1]

        converger = cls(cc, atThetas, atPhis)

        # Override history initialized in __init__ with CSV data.
        converger.allPhiSteps = list(phiSteps_mat)
        converger.allThetaSteps = list(thetaSteps_mat)
        converger.allPhis = list(phis_mat)
        converger.allThetas = list(thetas_mat)

        # Restore per-iteration tracking columns (one value per iteration, constant across cobras).
        if 'mcsFrameId' in df.columns:
            converger.allMcsFrameIds = df.groupby('iteration')['mcsFrameId'].first().to_list()
        if 'spsVisit' in df.columns:
            converger.allSpsVisits = df.groupby('iteration')['spsVisit'].first().to_list()
        if 'flux' in df.columns:
            converger.allFlux = list(pivot('flux'))
            converger.allFluxNorm = list(pivot('flux_norm')) if 'flux_norm' in df.columns else []
            converger.allFluxRatio = list(pivot('flux_ratio')) if 'flux_ratio' in df.columns else []
            converger.allAttenuationNorm = list(pivot('attenuation_norm')) if 'attenuation_norm' in df.columns else []

        # Targets and flags are constant per cobra — read from last iteration.
        # Rename legacy 'flags' column to 'cobraFlags' for back-compat.
        if 'flags' in df.columns:
            df = df.rename(columns={'flags': 'cobraFlags'})
        last = df[df.iteration == df.iteration.max()].sort_values('cobraId')
        converger.phiTarget = last.phiTarget.to_numpy()
        converger.thetaTarget = last.thetaTarget.to_numpy()
        converger.cobraFlags = last.cobraFlags.to_numpy().astype(int)

        if 'slewStartIter' in df.columns:
            converger.slewStartIter = last.slewStartIter.to_numpy().astype(int)

        if 'stepsToCenter' in df.columns:
            converger.stepsToCenter = last.stepsToCenter.to_numpy()
        else:
            converger.stepsToCenter = np.zeros(nCobras)

        return converger

    def dumpConvergenceData(self, DUMP_DIR=None):
        """Dump per-cobra per-iteration convergence history to a CSV for offline analysis.

        Columns: cobraId, iteration, phiSteps (that iteration), cumSteps, phi (rad), phiTarget (rad),
                 theta (rad), thetaTarget (rad), flags.
        Rows where phi/theta are nan correspond to iterations where the cobra was not detected by MCS.
        """
        nIter = len(self.allPhiSteps)
        nCobras = len(self.cc.allCobras)
        phiSteps = np.array(self.allPhiSteps)  # (nIter, nCobras)
        thetaSteps = np.array(self.allThetaSteps)  # (nIter, nCobras)
        phis = np.array(self.allPhis)  # (nIter, nCobras)
        thetas = np.array(self.allThetas)  # (nIter, nCobras)
        def _padOrFill(lst):
            """Convert list of arrays to (nIter, nCobras) matrix, padding with NaN if shorter."""
            if not lst:
                return np.full((nIter, nCobras), np.nan)
            arr = np.array(lst)
            if arr.shape[0] < nIter:
                pad = np.full((nIter - arr.shape[0], nCobras), np.nan)
                arr = np.vstack([arr, pad])
            return arr

        flux = _padOrFill(self.allFlux)
        fluxNorm = _padOrFill(self.allFluxNorm)
        fluxRatio = _padOrFill(self.allFluxRatio)
        attenuationNorm = _padOrFill(self.allAttenuationNorm)

        cobraIds = np.arange(1, nCobras + 1)

        stepsToCenter = self.stepsToCenter if self.stepsToCenter is not None else np.zeros(nCobras)

        frames = []
        for t in range(nIter):
            mcsFrameId = self.allMcsFrameIds[t] if t < len(self.allMcsFrameIds) else -1
            spsVisit = self.allSpsVisits[t] if t < len(self.allSpsVisits) else -1
            frames.append(pd.DataFrame({
                'cobraId': cobraIds,
                'iteration': t,
                'mcsFrameId': mcsFrameId,
                'spsVisit': spsVisit,
                'phiSteps': phiSteps[t],
                'thetaSteps': thetaSteps[t],
                'phi': phis[t],
                'theta': thetas[t],
                'flux': flux[t],
                'flux_norm': fluxNorm[t],
                'flux_ratio': fluxRatio[t],
                'attenuation_norm': attenuationNorm[t],
                'phiTarget': self.phiTarget,
                'thetaTarget': self.thetaTarget,
                'stepsToCenter': stepsToCenter,
                'cobraFlags': self.cobraFlags,
                'slewStartIter': self.slewStartIter,
            }))
        df = pd.concat(frames, ignore_index=True)

        visitStr = f'{self.visitId:06d}' if self.visitId is not None else 'unknown'
        DUMP_DIR = 'dotConvergence' if DUMP_DIR is None else DUMP_DIR
        outDir = os.path.join(ROOT_DIR, DUMP_DIR, f'v{visitStr}')
        os.makedirs(outDir, exist_ok=True)
        filepath = os.path.join(outDir, f'dotRoach_convergence_v{self.dumpVersion:02d}.csv')
        df.to_csv(filepath, index=False)
        self.dumpVersion += 1
        return filepath

    def _getCurrentSpsVisit(self):
        """Return the current SPS visit to use for flux retrieval."""
        return int(self.db.query_scalar('SELECT MAX(pfs_visit_id) FROM sps_visit'))


class ReplayDotConverger(DotConverger):
    """DotConverger subclass that replays historical data from the DB instead of live queries.

    Overrides _getCurrentSpsVisit so that SPS iterations can be driven
    one-by-one from a list of historical visit IDs (e.g. read from a
    previously dumped CSV) without touching live hardware.

    Usage
    -----
    csv = pd.read_csv(path)
    spsVisits = csv[csv.spsVisit != -1].groupby('iteration')['spsVisit'].first().tolist()
    converger = ReplayDotConverger(cc, atThetas, atPhis, spsVisits)
    """

    def __init__(self, cc, atThetas, atPhis, spsVisits):
        super().__init__(cc, atThetas, atPhis)
        self._spsVisitIter = iter(spsVisits)
        cc.setCurrentAngles(cc.allCobras, thetaAngles=atThetas, phiAngles=atPhis)

    def _getCurrentSpsVisit(self):
        """Return next historical SPS visit ID instead of querying MAX."""
        return next(self._spsVisitIter)
