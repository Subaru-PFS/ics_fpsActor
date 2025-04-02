import glob
import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import psycopg2
from pfs.datamodel import PfsConfig, TargetType
from pfs.utils.butler import Butler as Nestor
from pfs.utils.cobraMaskFile import buildCobraMaskFile
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
FIBER_BROKEN_MASK = (calibModel.status & calibModel.FIBER_BROKEN_MASK).astype('bool')
COBRA_OK_MASK = (calibModel.status & calibModel.COBRA_OK_MASK).astype('bool')

sgfm['x'] = xCob
sgfm['y'] = yCob
sgfm['FIBER_BROKEN_MASK'] = FIBER_BROKEN_MASK
sgfm['COBRA_OK_MASK'] = COBRA_OK_MASK
sgfm['armLength'] = armLength
sgfm['L1'] = L1
sgfm['L2'] = L2
# adding blackSpots position and radius.
np.testing.assert_equal(sgfm.cobraId.to_numpy(), dots.spotId.to_numpy())
sgfm['xDot'] = dots.x.to_numpy()
sgfm['yDot'] = dots.y.to_numpy()
sgfm['rDot'] = dots.r.to_numpy()

sgfm = sgfm[['scienceFiberId', 'cobraId', 'fiberId', 'spectrographId',
             'FIBER_BROKEN_MASK', 'COBRA_OK_MASK', 'x', 'y', 'xDot', 'yDot', 'rDot', 'armLength', 'L1', 'L2']]


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


def getLatestNearDotConvergenceId():
    nearDotConvergence = read_sql(
        "select * from iic_sequence inner join visit_set on visit_set.iic_sequence_id=iic_sequence.iic_sequence_id "
        "inner join pfs_config on pfs_config.visit0=visit_set.pfs_visit_id "
        "inner join pfs_design on pfs_design.pfs_design_id=pfs_config.pfs_design_id "
        "where sequence_type='nearDotConvergence' order by iic_sequence.iic_sequence_id desc")
    return nearDotConvergence.visit0.max()


def circleIntersections(x1, y1, r1, x2, y2, r2):
    """
    Calculate the intersection points of two circles.

    Parameters
    ----------
    x1, y1 : float
        Coordinates of the center of the first circle.
    r1 : float
        Radius of the first circle.
    x2, y2 : float
        Coordinates of the center of the second circle.
    r2 : float
        Radius of the second circle.

    Returns
    -------
    list of tuples
        A list of intersection points as (x, y) tuples. Returns an empty list if no intersections.
    """
    # Distance between the centers
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check if circles intersect or are coincident
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return []  # No intersection or circles are coincident

    # Calculate the length from the first circle's center to the intersection points
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(r1 ** 2 - a ** 2)

    # Midpoint between the intersections
    xMid = x1 + a * (x2 - x1) / d
    yMid = y1 + a * (y2 - y1) / d

    # Calculate the intersection points
    xInt1 = xMid + h * (y2 - y1) / d
    yInt1 = yMid - h * (x2 - x1) / d

    xInt2 = xMid - h * (y2 - y1) / d
    yInt2 = yMid + h * (x2 - x1) / d

    return [(xInt1, yInt1), (xInt2, yInt2)]


def findPhiCenter(cobraData):
    """
    Find the optimal intersection point (phi center) between two circles
    based on the smallest angular difference between home and target positions.

    Parameters
    ----------
    cobraData : object
        An object containing the cobra data with the following attributes:
        - x, y : Coordinates of the first circle's center.
        - L1 : Radius of the first circle.
        - xPosition, yPosition : Coordinates of the second circle's center.
        - L2 : Radius of the second circle.
        - xTarget, yTarget : Target position coordinates.

    Returns
    -------
    tuple
        The (x, y) coordinates of the optimal phi center.
    """
    # Get intersection points between the two circles
    intersections = circleIntersections(
        cobraData.x, cobraData.y, cobraData.L1,
        cobraData.xPosition, cobraData.yPosition, cobraData.L2
    )

    if not intersections:
        raise ValueError("No valid intersection points found between the circles.")

    # Compute angular differences for both intersections
    angleDiffs = []
    for cx, cy in intersections:
        angleHome = np.arctan2(cobraData.y - cy, cobraData.x - cx)
        angleTarget = np.arctan2(cobraData.yTarget - cy, cobraData.xTarget - cx)

        angleDiff = (angleTarget - angleHome) % (2 * np.pi)
        angleDiffs.append(angleDiff)

    # Select the intersection point with the smallest angular difference
    bestIndex = np.argmin(angleDiffs)
    return intersections[bestIndex]


def findNoCrossingCobras(nearConvergenceId):
    """
    Identify cobras that do not cross the black dot.

    Parameters:
    nearConvergenceId : int
        Identifier for the near convergence dataset.

    Returns:
    list
        List of cobra IDs that do not cross any intersection.
    """
    noCrossingCobras = []

    nearConvergence = loadConvergenceDf(nearConvergenceId)
    nearConvergence = nearConvergence[nearConvergence.COBRA_OK_MASK]

    for cobraId, cobraData in nearConvergence.groupby('cobraId'):
        cobraData = cobraData.squeeze()

        phiX, phiY = findPhiCenter(cobraData)
        interPhiDot = circleIntersections(phiX, phiY, cobraData.L2, cobraData.xDot, cobraData.yDot, cobraData.rDot)

        if not len(interPhiDot):
            noCrossingCobras.append(cobraId)

    return noCrossingCobras


def makeHideCobraMaskFile(cobraMatch, iteration, outputDir):
    """
    Create a mask file where only visible cobras moves.

    Parameters:
    cobraMatch : DataFrame
        Dataframe containing cobra match information.
    iteration : int
        Iteration number for filename reference.
    outputDir : str
        Directory where the mask file will be saved.

    Returns:
    tuple
        Tuple containing the mask file data and its file path.
    """
    cobraMatch = pd.merge(cobraMatch.rename(columns={'cobra_id': 'cobraId'}, inplace=False), sgfm, on='cobraId',
                          how='inner')
    doMove = cobraMatch.fiberId[cobraMatch.spot_id != -1].to_numpy()

    fileName = f'{iteration:02d}'
    maskFile = buildCobraMaskFile(doMove, fileName, doSave=True, outputDir=outputDir)
    maskFilepath = os.path.join(outputDir, f'{fileName}.csv')

    return maskFile, maskFilepath


def calcSpeed(cobraData, scalingDf):
    """
    Calculate the angular speed of a cobra.

    Parameters:
    cobraData : DataFrame
        Data for a specific cobra.
    scalingDf : DataFrame
        Scaling data containing historical cobra positions.
    nIterForScaling : int
        Number of iterations to consider for scaling.

    Returns:
    float
        Median angular speed of the cobra.
    """

    def robustFindPhiCenter(cobraData, dfi2):
        robustPhi = []

        for iteration in range(-1, dfi2.iteration.max() + 1):
            cobraData = cobraData.copy()

            if iteration != -1:
                iterVal = dfi2[dfi2.iteration == iteration].squeeze()
                cobraData['xPosition'] = iterVal.pfi_center_x_mm
                cobraData['yPosition'] = iterVal.pfi_center_y_mm

                try:
                    robustPhi.append(findPhiCenter(cobraData))
                except:
                    robustPhi.append((np.nan, np.nan))

        robustPhi = np.array(robustPhi)
        return np.nanmedian(robustPhi, axis=0)

    def calcAngularSpeed(dx, dy):
        angles = np.arctan2(dy, dx)  # Angles in radians
        speed = np.diff(angles)

        return np.median(speed)

    dfi2 = scalingDf[scalingDf.cobra_id == cobraData.cobraId].sort_values('iteration')
    dfi2.loc[dfi2.spot_id == -1, 'pfi_center_x_mm'] = np.nan
    dfi2.loc[dfi2.spot_id == -1, 'pfi_center_y_mm'] = np.nan

    phiX, phiY = robustFindPhiCenter(cobraData, dfi2)

    xMm = np.concatenate([[cobraData.xPosition], dfi2.pfi_center_x_mm.to_numpy()])
    yMm = np.concatenate([[cobraData.yPosition], dfi2.pfi_center_y_mm.to_numpy()])

    iEnd = len(xMm) // 2 + 1
    useX1, useX2 = xMm[:iEnd], xMm[iEnd - 1:]
    useY1, useY2 = yMm[:iEnd], yMm[iEnd - 1:]

    speed1 = calcAngularSpeed(useX1 - phiX, useY1 - phiY)
    speed2 = calcAngularSpeed(useX2 - phiX, useY2 - phiY)

    interPhiDot = np.array(
        circleIntersections(phiX, phiY, cobraData.L2, cobraData.xDot, cobraData.yDot, cobraData.rDot / 2))

    if interPhiDot.size:
        interX, interY = interPhiDot[np.argmin(np.hypot(interPhiDot[:, 0] - useX2[-1], interPhiDot[:, 1] - useY2[-1]))]
        distance = np.arctan2(interY - phiY, interX - phiX)  # Angles in radians
    else:
        distance = np.nan

    return speed1, speed2, distance


def process_cobra(args):
    """
    Worker function to process a single cobraId group.
    """
    cobraId, cobraData, scalingDf = args
    cobraData = cobraData.squeeze()

    try:
        speed1, speed2, distance = calcSpeed(cobraData, scalingDf)
    except:
        speed1, speed2, distance = np.nan, np.nan, np.nan

    return cobraId, speed1, speed2, distance


def parallel_speed_calculation(convergenceDf, scalingDf, num_cores=2):
    """
    Perform parallel computation of speeds and scalings for cobras.

    Parameters:
    convergenceDf : DataFrame
        Dataframe containing convergence information.
    scalingDf : DataFrame
        Dataframe containing scaling information.
    num_cores : int
        Number of cores to use for multiprocessing.

    Returns:
    DataFrame
        Dataframe with calculated speeds and scalings.
    """
    # Filter valid cobras
    convergenceDf = convergenceDf[convergenceDf.COBRA_OK_MASK]

    results = [process_cobra((cobraId, group, scalingDf)) for cobraId, group in convergenceDf.groupby('cobraId')]

    # Prepare arguments for multiprocessing
    # cobra_groups = [(cobraId, group, scalingDf) for cobraId, group in convergenceDf.groupby('cobraId')]

    # Use multiprocessing to process cobras in parallel
    # with Pool(processes=num_cores) as pool:
    #    results = pool.map(process_cobra, cobra_groups)

    # Convert results to a DataFrame
    speeds = pd.DataFrame(results, columns=['cobraId', 'speed1', 'speed2', 'distance'])

    # Calculate scaling factors
    speeds['scaling1'] = speeds.speed1.median() / speeds.speed1.to_numpy()
    speeds['scaling2'] = speeds.speed2.median() / speeds.speed2.to_numpy()

    return speeds


def makeScaling(nearConvergenceId, visit, outputDir, maxScaling=5, minScaling=0.1, numCores=2):
    """
    Generate scaling factors for cobra speeds and save to a scaling file.

    Parameters:
    convergenceDf : DataFrame
        Dataframe containing convergence information.
    scalingDf : DataFrame
        Dataframe containing scaling data.
    nIterForScaling : int
        Number of iterations to consider for scaling calculations.
    outputDir : str
        Directory where the scaling file will be saved.
    maxScaling : float, optional
        Maximum scaling factor (default is 5).
    minScaling : float, optional
        Minimum scaling factor (default is 0.1).

    Returns:
    str
        File path of the generated scaling file.
    """
    # calculate the scaling
    convergenceDf = loadConvergenceDf(nearConvergenceId)
    scalingDf = getCobraMatchData(visit)

    speeds = parallel_speed_calculation(convergenceDf, scalingDf, num_cores=numCores)

    # Calculate scaling factors
    speeds['scaling1'] = speeds.speed1.median() / speeds.speed1.to_numpy()
    speeds['scaling2'] = speeds.speed2.median() / speeds.speed2.to_numpy()

    for column in ['scaling1', 'scaling2']:
        speeds.loc[np.isnan(speeds.scaling2), column] = 1
        speeds.loc[speeds.scaling2 < 0, column] = 1
        speeds.loc[speeds.scaling2 > 10, column] = 1

        speeds.loc[speeds[column] > maxScaling, column] = maxScaling
        speeds.loc[speeds[column] < minScaling, column] = minScaling

    final = pd.DataFrame(dict(cobraId=np.arange(1, 2395)))

    for column in speeds.columns:
        if column == 'cobraId':
            continue
        defaultVal = 1 if 'scaling' in column else np.nan
        final[column] = defaultVal
        final.loc[speeds.cobraId.to_numpy() - 1, column] = speeds[column].to_numpy()

    filepath = os.path.join(outputDir, 'scaling.csv')
    final.to_csv(filepath)

    return filepath
