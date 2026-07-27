import glob
from datetime import timedelta

import numpy as np
import pandas as pd
import pfs.utils.coordinates.transform as transformUtils
import psycopg2
from pfs.datamodel import PfsConfig, TargetType
from pfs.utils.butler import Butler as Nestor
from pfs.utils.fiberids import FiberIds

nestor = Nestor()

dots = nestor.get('black_dots')
calibModel = nestor.get('moduleXml', moduleName='ALL', version='')

gfm = pd.DataFrame(FiberIds().data)
sgfm = gfm.set_index('scienceFiberId').loc[np.arange(2394) + 1].reset_index().sort_values('cobraId')

# getting up-to-date cobras calibration.
xCob = np.array(calibModel.centers.real).astype('float32')
yCob = np.array(calibModel.centers.imag).astype('float32')
armLength = np.array(calibModel.L1 + calibModel.L2).astype('float32')
L1 = np.array(calibModel.L1).astype('float32')
L2 = np.array(calibModel.L2).astype('float32')
tht0 = np.array(calibModel.tht0).astype('float32')
FIBER_BROKEN_MASK = (calibModel.status & calibModel.FIBER_BROKEN_MASK).astype('bool')
COBRA_OK_MASK = (calibModel.status & calibModel.COBRA_OK_MASK).astype('bool')

sgfm['x'] = xCob
sgfm['y'] = yCob
sgfm['FIBER_BROKEN_MASK'] = FIBER_BROKEN_MASK
sgfm['COBRA_OK_MASK'] = COBRA_OK_MASK
sgfm['armLength'] = armLength
sgfm['L1'] = L1
sgfm['L2'] = L2
sgfm['tht0'] = tht0
# adding blackSpots position and radius.
np.testing.assert_equal(sgfm.cobraId.to_numpy(), dots.spotId.to_numpy())
sgfm['xDot'] = dots.x.to_numpy()
sgfm['yDot'] = dots.y.to_numpy()
sgfm['rDot'] = dots.r.to_numpy()

sgfm = sgfm[['scienceFiberId', 'cobraId', 'fiberId', 'spectrographId',
             'FIBER_BROKEN_MASK', 'COBRA_OK_MASK', 'x', 'y', 'xDot', 'yDot', 'rDot', 'armLength', 'L1', 'L2', 'tht0']]


def getConn():
    """
    Establishes a connection to the PostgreSQL database 'opdb' on host 'pfsa-db01' and port 5432 with user 'pfs'.

    Returns:
    conn: A PostgreSQL connection object.
    """
    return psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'")


def read_sql(sql):
    """
    Executes a SQL query and returns the result as a DataFrame.

    Args:
    sql (str): SQL query to be executed.

    Returns:
    df: A pandas DataFrame containing the result of the SQL query.
    """
    with getConn() as conn:
        df = pd.read_sql(sql, conn)
        return df.loc[:, ~df.columns.duplicated()]


def robustRms(array, axis=None):
    """
    Calculates the robust Root Mean Square (RMS) of an array using the inter-quartile range.

    Args:
    array (numpy.ndarray): Input array.

    Returns:
    rms (float): Robust RMS of the input array.
    """
    lq, uq = np.nanpercentile(array, (25.0, 75.0), axis=axis)
    return 0.741 * (uq - lq)


def getCobraMatchData(visit, iteration=None, **kwargs):
    """
    Retrieve cobra match data from the database for a specific visit.

    Parameters
    ----------
    visit : int
        The PFS visit ID for which the cobra match data should be retrieved.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the following columns:
        - pfs_visit_id: The PFS visit ID.
        - iteration: The iteration number of the match.
        - cobra_id: The cobra identifier.
        - spot_id: The spot identifier from the match data.
        - pfi_center_x_mm: X-coordinate of the spot in PFI in millimeters.
        - pfi_center_y_mm: Y-coordinate of the spot in PFI in millimeters.
    """
    sql = (
        'SELECT pfs_visit_id, iteration, cobra_id, cobra_match.spot_id, pfi_center_x_mm, pfi_center_y_mm, mcs_center_x_pix, mcs_center_y_pix '
        'FROM mcs_data '
        'LEFT OUTER JOIN cobra_match '
        'ON mcs_data.spot_id = cobra_match.spot_id '
        'AND mcs_data.mcs_frame_id = cobra_match.mcs_frame_id '
        f'WHERE cobra_match.pfs_visit_id = {visit} '
        'ORDER BY cobra_id ASC'
    )

    allIterations = read_sql(sql)
    selected = allIterations if iteration is None else allIterations[allIterations.iteration == iteration]

    return selected


def fetchPfsDesignId(visit0):
    """
    Fetch the pfsDesignId and design name associated with the given visit0.

    Parameters
    ----------
    visit0 : int
        The visit0 identifier.

    Returns
    -------
    designId : int
        The design ID associated with the given visit0.
    """
    query = (
        f"SELECT pfs_config.pfs_design_id, design_name "
        f"FROM pfs_config "
        f"INNER JOIN pfs_design ON pfs_design.pfs_design_id = pfs_config.pfs_design_id "
        f"WHERE visit0 = {visit0}"
    )
    designId, designName = read_sql(query).squeeze().to_numpy()
    print(f'pfsConfig-0x{designId:016x}-{visit0:06d} : {designName}')
    return designId


def fetchDateDir(visit0):
    """
    Fetch the possible date directories associated with the given visit0.

    Parameters
    ----------
    visit0 : int
        The visit0 identifier.

    Returns
    -------
    dates : list of str
        List of possible date directories in 'YYYY-MM-DD' format.
    """

    def getFormattedDate(date, deltaDays=0):
        """Return the date in 'YYYY-MM-DD' format, with an optional day shift."""
        adjustedDate = date + timedelta(days=deltaDays)
        return adjustedDate.strftime('%Y-%m-%d')

    query = f"SELECT issued_at FROM pfs_visit WHERE pfs_visit_id = {visit0}"
    issuedAt = read_sql(query).squeeze()

    # Generate possible date directories: today and the next day
    dates = [getFormattedDate(issuedAt, deltaDays=+1), getFormattedDate(issuedAt)]
    return dates


def loadPfsConfig0(visit0, skipEngineering=True):
    """
    Load the pfsConfig for the given visit0.

    Parameters
    ----------
    visit0 : int
        The visit0 identifier.
    skipEngineering : bool, optional
        Whether to skip engineering fibers (default: True).

    Returns
    -------
    pfsConfig : PfsConfig
        The loaded PfsConfig object.

    Raises
    ------
    RuntimeError
        If no matching pfsConfig file is found.
    """
    designId = fetchPfsDesignId(visit0)
    dateDirs = fetchDateDir(visit0)

    # Search for the pfsConfig file in the possible date directories
    for date in dateDirs:
        configPath = glob.glob(f'/data/raw/{date}/pfsConfig/pfsConfig-0x{designId:016x}-{visit0:06d}.fits')
        if configPath:
            break

    if not configPath:
        raise RuntimeError(f'Could not find matching pfsConfig0 for visit0: {visit0}')

    print(f'Reading pfsConfig from {configPath[0]}')
    pfsConfig = PfsConfig._readImpl(configPath[0])

    # Optionally skip engineering fibers
    if skipEngineering:
        pfsConfig = pfsConfig[pfsConfig.targetType != TargetType.ENGINEERING]

    return pfsConfig


def loadConvergenceDf(visit0):
    """
    Load the convergence data and PFS configuration into a DataFrame.

    Parameters
    ----------
    visit0 : int
        The visit0 identifier.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing convergence data and PFS configuration.
    """
    # Load the science configuration for the given visit0
    scienceConfig = loadPfsConfig0(visit0, skipEngineering=True)

    # Create a DataFrame with the necessary fields from the PFS configuration
    df = pd.DataFrame({
        'fiberId': scienceConfig.fiberId.astype('int32'),
        'spectrograph': scienceConfig.spectrograph.astype('int32'),
        'fiberStatus': scienceConfig.fiberStatus.astype('int32'),
        'targetType': scienceConfig.targetType.astype('int32'),
        'xTarget': scienceConfig.pfiNominal[:, 0].astype('float32'),
        'yTarget': scienceConfig.pfiNominal[:, 1].astype('float32'),
        'xPosition': scienceConfig.pfiCenter[:, 0].astype('float32'),
        'yPosition': scienceConfig.pfiCenter[:, 1].astype('float32')
    })

    # Add metadata to the DataFrame
    df['designId'] = scienceConfig.pfsDesignId
    df['visit'] = scienceConfig.visit
    df['designName'] = scienceConfig.designName

    # Calculate the displacement between nominal and actual positions
    diff = (scienceConfig.pfiCenter - scienceConfig.pfiNominal).astype('float32')
    df['dx'] = diff[:, 0]
    df['dy'] = diff[:, 1]
    df['dist'] = np.hypot(diff[:, 0], diff[:, 1])

    # Merge with additional data from sgfm
    return pd.merge(df, sgfm, on='fiberId', how='inner')


def getMcsData(**kwargs):
    """Retrieve final cobra position in mm.

    Parameters
    ----------
    visitId : `int`
        Convergence identifier.
    """
    mcsIds = McsVisitId(**kwargs)
    sql = f'SELECT * from mcs_data inner join mcs_exposure on mcs_exposure.mcs_frame_id=mcs_data.mcs_frame_id where mcs_data.mcs_frame_id={mcsIds.mcsFrameId}'
    return read_sql(sql)


class McsVisitId(dict):
    """
    A dictionary-like class for representing an MCS visit ID.

    Attributes:
        visit (int): The visit number.
        iteration (int): The iteration number.
        mcsFrameId (int): The MCS frame ID.

    Methods:
        visit: Get the visit number.
        iteration: Get the iteration number.
        mcsFrameId: Get the MCS frame ID.
    """

    def __init__(self, visit=0, iteration=0, mcsFrameId=0):
        """
        Initialize an McsVisitId object.

        Parameters:
            visit (int): The visit number.
            iteration (int): The iteration number.
            mcsFrameId (int): The MCS frame ID.
        """

        if mcsFrameId:
            visit = int(mcsFrameId / 100)
            iteration = mcsFrameId - 100 * visit

        else:
            mcsFrameId = 100 * visit + iteration

        self['visit'] = visit
        self['iteration'] = iteration
        self['mcsFrameId'] = mcsFrameId

    @property
    def visit(self):
        """int: The visit number."""
        return self['visit']

    @property
    def iteration(self):
        """int: The iteration number."""
        return self['iteration']

    @property
    def mcsFrameId(self):
        """int: The MCS frame ID."""
        return self['mcsFrameId']


def getMcsPfiTransform(mcs_frame_id):
    sql = f'SELECT * from mcs_pfi_transformation where mcs_frame_id={mcs_frame_id}'
    return read_sql(sql)


def constructMcsPfiTransform(**kwargs):
    mcsIds = McsVisitId(**kwargs)
    allSpots = getMcsData(**kwargs)
    altitude, = allSpots.altitude.unique()
    insrot, = allSpots.insrot.unique()

    param = getMcsPfiTransform(mcsIds.mcsFrameId)
    camera_name = param.squeeze().camera_name
    camera_name = 'usmcs' if camera_name == 'rmod_71m' else camera_name
    pfiTransform = transformUtils.fromCameraName(camera_name, altitude=altitude, insrot=insrot)
    pfiTransform.mcsDistort.setArgs(*param[['x0', 'y0', 'theta', 'dscale', 'scale2']].to_numpy())

    return pfiTransform


def getMcsDataOnPfi(mcsVisit, iteration=0):
    mcsFrameId = 100 * mcsVisit + iteration

    allSpots = getMcsData(mcsFrameId=mcsFrameId)
    pfiTransform = constructMcsPfiTransform(mcsFrameId=mcsFrameId)

    x_mm, y_mm = pfiTransform.mcsToPfi(allSpots['mcs_center_x_pix'].to_numpy(), allSpots['mcs_center_y_pix'].to_numpy())
    allSpots['pfi_center_x_mm'] = x_mm
    allSpots['pfi_center_y_mm'] = y_mm

    return allSpots
