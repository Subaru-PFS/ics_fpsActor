"""Measure the MCS boresight from opdb frames, with pfs.utils.coordinates.boresight, and record it."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pfs.utils.coordinates.boresight import measureBoresight

MAX_UNCERTAINTY = 0.25
"""Largest uncertainty of a boresight, pixels per axis, written to opdb."""

POINTS_SQL = """
SELECT m.mcs_frame_id, 'cobra' AS kind, m.cobra_id AS id, d.mcs_center_x_pix AS x, d.mcs_center_y_pix AS y
FROM cobra_match m JOIN mcs_data d ON d.mcs_frame_id = m.mcs_frame_id AND d.spot_id = m.spot_id
WHERE m.mcs_frame_id = ANY(:frameIds) AND m.spot_id >= 0
UNION ALL
SELECT m.mcs_frame_id, 'fiducial' AS kind, m.fiducial_fiber_id AS id, d.mcs_center_x_pix, d.mcs_center_y_pix
FROM fiducial_fiber_match m JOIN mcs_data d ON d.mcs_frame_id = m.mcs_frame_id AND d.spot_id = m.spot_id
WHERE m.mcs_frame_id = ANY(:frameIds) AND m.spot_id >= 0
"""


def loadFrames(db, frameIds):
    """Load the frames' telescope state and the spots fibre identification named.

    Parameters
    ----------
    db : `pfs.utils.database.db.DB`
    frameIds : sequence of `int`
        mcs_frame_id of the frames, distinct.

    Returns
    -------
    frames : `pandas.DataFrame`
        mcs_frame_id, insrot, altitude, in the order of frameIds.
    points : `pandas.DataFrame`
        mcs_frame_id, kind ('cobra' or 'fiducial'), id, x, y (MCS pixels), finite only.

    Raises
    ------
    ValueError
        A frame is given twice, or has no mcs_exposure row.
    """
    frameIds = [int(frameId) for frameId in frameIds]
    if len(set(frameIds)) != len(frameIds):
        raise ValueError(f'frames given more than once: {frameIds}')

    params = dict(frameIds=frameIds)
    frames = db.query_dataframe(
        'SELECT mcs_frame_id, insrot, altitude FROM mcs_exposure ' 'WHERE mcs_frame_id = ANY(:frameIds)',
        params=params,
    )
    missing = sorted(set(frameIds) - set(frames.mcs_frame_id))
    if missing:
        raise ValueError(f'no mcs_exposure for frames {missing}')

    frames = frames.set_index('mcs_frame_id').loc[frameIds].reset_index()
    points = db.query_dataframe(POINTS_SQL, params=params)
    # a NULL centroid can make the column object.
    points[['x', 'y']] = points[['x', 'y']].astype(float)
    points = points[np.isfinite(points.x) & np.isfinite(points.y)].reset_index(drop=True)
    return frames, points


def calcBoresight(cmd, db, frameIds, pfsVisitId, writeToDB=True):
    """Measure the boresight from frameIds and report it; write it to opdb if writeToDB.

    Parameters
    ----------
    cmd : `actorcore.Command`
        Receives the report, frame by frame.
    db : `pfs.utils.database.db.DB`
    frameIds : sequence of `int`
        mcs_frame_id of the frames.
    pfsVisitId : `int`
        Visit the measurement is recorded under.
    writeToDB : `bool`
        Write the measurement to mcs_boresight_meas and its frames to mcs_boresight_frame_set.

    Returns
    -------
    `pfs.utils.coordinates.boresight.BoresightResult`

    Raises
    ------
    ValueError
        The frames cannot be measured; or writeToDB, and a frame is still displaced from the
        others or the uncertainty exceeds MAX_UNCERTAINTY.
    """
    frames, points = loadFrames(db, frameIds)
    result = measureBoresight(frames, points)

    for row in result.frames.itertuples():
        status = 'used' if row.used else f'left out: {row.reason}'
        rotation = (row.rotation + 180) % 360 - 180
        cmd.inform(
            f'text="frame {row.mcs_frame_id}: insrot={row.insrot:.2f} rotation={rotation:.3f} '
            f'offset={row.rotationOffset:+.3f} deg, {row.nPoints} spots ({row.nLeftOut} left out), '
            f'median residual {row.residual:.3f} pix, displacement {row.displacement:.3f} pix, {status}"'
        )

    sigmaF = 'measured' if result.sigmaFMeasured else 'assumed'
    cmd.inform(
        f'text="Boresight {result.x:.3f}, {result.y:.3f} +- {result.xErr:.3f}, {result.yErr:.3f} pix '
        f'from {len(result.framesUsed)} frames at {result.nPositions} rotator positions, altitude '
        f'{result.altitude:.2f}; frame displacement {result.sigmaF:.3f} pix ({sigmaF}); {result.nPoints} '
        f'spots, {result.nLeftOut} left out."'
    )
    for warning in result.warnings:
        cmd.warn(f'text="{warning}"')

    if writeToDB is not True:
        cmd.inform('text="Not writing to DB."')
        return result

    if result.standingOut is not None:
        raise ValueError(f'frame {result.standingOut} is still displaced from the other frames: not written')
    if max(result.xErr, result.yErr) > MAX_UNCERTAINTY:
        why = (
            f'its {result.nPositions} rotator positions span too little rotation'
            if not result.sigmaFMeasured
            else f'its frames are displaced by {result.sigmaF:.3f} pix rms'
        )
        raise ValueError(
            f'boresight uncertainty {result.xErr:.3f}, {result.yErr:.3f} pix exceeds '
            f'{MAX_UNCERTAINTY} pix because {why}: not written'
        )

    writeBoresightToDB(db, pfsVisitId, result)
    return result


def writeBoresightToDB(db, pfsVisitId, result):
    """write a measured boresight to mcs_boresight_meas with the current timestamp,
    and the frames it was measured from to mcs_boresight_frame_set"""

    frameSetId = db.query_scalar("SELECT nextval('mcs_boresight_frame_set_id_seq')")
    db.insert(
        'mcs_boresight_frame_set',
        pd.DataFrame({'mcs_boresight_frame_set_id': frameSetId, 'mcs_frame_id': result.framesUsed}),
    )

    dt = datetime.now(timezone.utc)

    df = pd.DataFrame(
        {
            'pfs_visit_id': [pfsVisitId],
            'mcs_boresight_x_pix': [result.x],
            'mcs_boresight_y_pix': [result.y],
            'mcs_boresight_x_err_pix': [result.xErr],
            'mcs_boresight_y_err_pix': [result.yErr],
            'altitude': [result.altitude],
            'calculated_at': [dt],
            'mcs_boresight_frame_set_id': [frameSetId],
        }
    )

    db.insert('mcs_boresight_meas', df)
