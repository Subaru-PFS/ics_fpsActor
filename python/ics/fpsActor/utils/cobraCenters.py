import numpy as np
import pandas as pd
import pfs.utils.coordinates.transform as transformUtils
from ics.fpsActor.utils.alfUtils import sgfm, nestor
from scipy.spatial import cKDTree

RADIUS = dict(COBRA_OK=1.0,
              COBRA_BROKEN=8.0,
              FIDUCIAL=0.2)


def match_spots_exclusive(spotPositions, spotId, allPositions, allRadius, allIds):
    """
    Match MCS spots to cobras/fiducials with exclusive one-to-one assignment.
    Uses a closest-first greedy approach to resolve conflicts.

    Parameters
    ----------
    spotPositions : (N, 2) array — spot x/y positions in PFI frame
    spotId        : (N,)   array — MCS spot IDs
    allPositions  : (M, 2) array — cobra + fiducial x/y positions
    allRadius     : (M,)   array — match radius per object
    allIds        : (M,)   array — cobra IDs (positive) + fiducial IDs (negative)

    Returns
    -------
    matchedSpotId : matched spot IDs
    matchedObjId  : corresponding object IDs (from allIds)
    matchedDist   : match distances
    """
    tree = cKDTree(allPositions)
    rMax = allRadius.max()
    candIdxList = tree.query_ball_point(spotPositions, r=rMax)

    candidates = []
    for iSpot, candIdx in enumerate(candIdxList):
        if len(candIdx) == 0:
            continue

        candIdx = np.asarray(candIdx)
        dx = allPositions[candIdx, 0] - spotPositions[iSpot, 0]
        dy = allPositions[candIdx, 1] - spotPositions[iSpot, 1]
        dist = np.hypot(dx, dy)

        keep = dist <= allRadius[candIdx]
        if not np.any(keep):
            continue

        candIdx = candIdx[keep]
        dist = dist[keep]
        iBest = np.argmin(dist)
        candidates.append((dist[iBest], iSpot, candIdx[iBest]))

    # Sort by distance — closest pair gets priority
    candidates.sort(key=lambda x: x[0])

    seenSpot = set()
    seenObj = set()
    matchedSpotId, matchedObjId, matchedDist = [], [], []

    for dist, iSpot, objIdx in candidates:
        if iSpot in seenSpot or objIdx in seenObj:
            continue
        seenSpot.add(iSpot)
        seenObj.add(objIdx)
        matchedSpotId.append(spotId[iSpot])
        matchedObjId.append(allIds[objIdx])
        matchedDist.append(dist)

    return np.array(matchedSpotId), np.array(matchedObjId), np.array(matchedDist)


def _buildMatchInputs():
    """Build cobra + fiducial positions/radii/IDs for spot matching.

    Non-OK (but non-broken) cobras get a larger search radius (8 mm) to account
    for their unknown position drift.
    """
    cobraMask = ~sgfm.FIBER_BROKEN_MASK.to_numpy()

    cobraRadius = np.full(len(sgfm), RADIUS['COBRA_OK'])
    cobraRadius[~sgfm.COBRA_OK_MASK.to_numpy()] = RADIUS['COBRA_BROKEN']

    cobraPositions = sgfm.loc[cobraMask, ["x", "y"]].to_numpy()
    cobraRadius = cobraRadius[cobraMask]
    cobraId = sgfm.cobraId.to_numpy()[cobraMask]

    fiducials = nestor.get('fiducials')
    fiducialPositions = fiducials[["x_mm", "y_mm"]].to_numpy()
    fidMask = np.isfinite(fiducialPositions[:, 0])
    fiducialPositions = fiducialPositions[fidMask]
    fiducialId = fiducials.fiducialId[fidMask].to_numpy() * -1
    fiducialRadius = np.full(len(fiducialPositions), RADIUS['FIDUCIAL'])

    allPositions = np.vstack([cobraPositions, fiducialPositions])
    allRadius = np.concatenate([cobraRadius, fiducialRadius])
    allIds = np.concatenate([cobraId, fiducialId])

    return allPositions, allRadius, allIds


def _getMcsDataOnPfi(db, mcsFrameId):
    """Fetch MCS spot data and transform pixel coordinates to PFI frame."""
    allSpots = db.query_dataframe(
        "SELECT * FROM mcs_data "
        "INNER JOIN mcs_exposure ON mcs_exposure.mcs_frame_id=mcs_data.mcs_frame_id "
        f"WHERE mcs_data.mcs_frame_id={mcsFrameId}"
    )
    altitude, = allSpots.altitude.unique()
    insrot, = allSpots.insrot.unique()

    param = db.query_dataframe(f"SELECT * FROM mcs_pfi_transformation WHERE mcs_frame_id={mcsFrameId}")
    camera_name = param.squeeze().camera_name
    camera_name = 'usmcs' if camera_name == 'rmod_71m' else camera_name
    pfiTransform = transformUtils.fromCameraName(camera_name, altitude=altitude, insrot=insrot)
    pfiTransform.mcsDistort.setArgs(*param[['x0', 'y0', 'theta', 'dscale', 'scale2']].to_numpy())

    x_mm, y_mm = pfiTransform.mcsToPfi(allSpots.mcs_center_x_pix.to_numpy(),
                                       allSpots.mcs_center_y_pix.to_numpy())
    allSpots['pfi_center_x_mm'] = x_mm
    allSpots['pfi_center_y_mm'] = y_mm

    return allSpots


def _fetchLastMoveToHomeVisit(db):
    """Return the pfs_visit_id of the last moveToHome sequence."""
    return db.query_dataframe(
        "SELECT max(pfs_visit_id) FROM iic_sequence "
        "INNER JOIN visit_set ON visit_set.iic_sequence_id=iic_sequence.iic_sequence_id "
        "WHERE sequence_type='moveToHome'"
    ).squeeze()


def _tipToCenter(calibModel, cobraIdx, tipPos):
    """Derive cobra center from observed tip position at home (theta=tht1, phi=phiIn).

    At home the theta arm is at the CCW hard stop (tht1) and the phi arm is
    folded at its CW hard stop (phiAngles=0, reference angle phiIn), so:

        ang1 = tht1
        ang2 = tht1 + phiIn
        tip  = center + L1*exp(j*ang1) + L2*exp(j*ang2)
    =>  center = tip - L1*exp(j*ang1) - L2*exp(j*ang2)

    Parameters
    ----------
    calibModel : PFIDesign
    cobraIdx   : (N,) int array — 0-based cobra indices
    tipPos     : (N,) complex array — observed tip positions in PFI mm

    Returns
    -------
    (N,) complex array — derived center positions
    """
    ang1 = calibModel.tht1[cobraIdx]
    ang2 = ang1 + calibModel.phiIn[cobraIdx]
    offset = calibModel.L1[cobraIdx] * np.exp(1j * ang1) + calibModel.L2[cobraIdx] * np.exp(1j * ang2)
    return tipPos - offset


def updateCobraCenters(calibModel, db, brokenOnly=True):
    """Update calibModel.centers using the last moveToHome MCS frame.

    For good (OK) cobras (when brokenOnly=False): derives the true center from
    the observed tip position using forward-kinematics geometry at home
    (theta=tht1, phi=phiIn).
    For non-OK cobras: sets the center directly to the observed spot position
    (angles unknown, geometry correction not possible).

    Parameters
    ----------
    calibModel : PFIDesign
        Live calibration model (e.g. self.cc.calibModel). Modified in-place.
    db : OpDB
        Active database connection (e.g. from self.connectToDB(cmd)).
    brokenOnly : bool, optional
        If True (default), only update non-OK cobra centers.
        If False, also update good cobra centers using geometry.

    Returns
    -------
    updatedCobraId : np.ndarray
        1-based cobra IDs whose centers were updated.
    isGood : np.ndarray of bool
        True for OK cobras, False for non-OK cobras.
    oldX, oldY : np.ndarray
        Previous PFI center x/y positions in mm.
    newX, newY : np.ndarray
        Updated PFI center x/y positions in mm.
    """
    homeVisit = _fetchLastMoveToHomeVisit(db)

    allSpots = _getMcsDataOnPfi(db, mcsFrameId=100 * homeVisit)
    allSpots = allSpots[allSpots.spot_id != -1].copy()

    mask = np.isfinite(allSpots.pfi_center_x_mm) & np.isfinite(allSpots.pfi_center_y_mm)
    allSpots = allSpots[mask]

    spotPositions = allSpots[['pfi_center_x_mm', 'pfi_center_y_mm']].to_numpy()
    spotId = allSpots.spot_id.to_numpy()

    allPositions, allRadius, allIds = _buildMatchInputs()

    matchedSpotId, matchedObjId, _ = match_spots_exclusive(
        spotPositions, spotId, allPositions, allRadius, allIds)

    # Keep only cobra matches (positive IDs) and join with cobra status
    cobMask = matchedObjId > 0
    matchDf = pd.DataFrame({
        'cobraId': matchedObjId[cobMask],
        'spot_id': matchedSpotId[cobMask],
    })
    matchDf = pd.merge(matchDf, sgfm[['cobraId', 'COBRA_OK_MASK']], on='cobraId')

    if brokenOnly:
        matchDf = matchDf[~matchDf.COBRA_OK_MASK]

    spotPos = allSpots.set_index('spot_id')[['pfi_center_x_mm', 'pfi_center_y_mm']]
    matchDf = matchDf.join(spotPos, on='spot_id')
    tipPos = matchDf.pfi_center_x_mm.to_numpy() + 1j * matchDf.pfi_center_y_mm.to_numpy()

    cobraIdx = matchDf.cobraId.to_numpy() - 1  # 0-based
    oldCenters = calibModel.centers[cobraIdx].copy()

    goodMask = matchDf.COBRA_OK_MASK.to_numpy()

    # Good cobras: derive center from geometry
    if goodMask.any():
        calibModel.centers[cobraIdx[goodMask]] = _tipToCenter(calibModel, cobraIdx[goodMask], tipPos[goodMask])

    # Non-OK cobras: angles unknown, use spot position directly
    brokenMask = ~goodMask
    if brokenMask.any():
        calibModel.centers[cobraIdx[brokenMask]] = tipPos[brokenMask]

    newCenters = calibModel.centers[cobraIdx]
    return matchDf.cobraId.to_numpy(), goodMask, oldCenters.real, oldCenters.imag, newCenters.real, newCenters.imag
