import numpy as np
import pandas as pd
from scipy import optimize
from datetime import datetime, timezone
from scipy.spatial import cKDTree
import cv2


def nearestNeighbourMatchingBore(points, targets, unrot):
    
    """
    simple matching for fiducial fibres, for use in home position

    input
       points: set to match (nx3)
       targets: set to match to (nx3)
       nTarg: length of targets

    """

    nTarg = points.shape[0]

    # use cKDTree for speed
    pointTree = cKDTree(points[:, 1:3])
    matchPoint = np.zeros((len(targets), 3))
    
    for i in range(len(targets)):
        dd, ii = pointTree.query(targets[i, 1:3], k = 1)
        matchPoint[i] = unrot[ii]

    return matchPoint

def distancePointLine(p1, p2, p):
    """ get closest distance from a line between two points and another point """

    y1 = p1[1]
    x1 = p1[0]
    y2 = p2[1]
    x2 = p2[0]
    x = p[0]
    y = p[1]

    d = abs((y2-y1)*x+(x2-x1)*y+x2*y1-y2*x1)/np.sqrt((y2-y1)**2+(x2-x1)**2)

    return p


def extractRotCent(afCoeff):    
    """extract centre of rotation from affine matrix"""
        
    #affine matrix is of the form
    #alpha   beta   (1-alpha)*xc-beta*yc
    #-beta   alpha  beta*xc-(1-alpha)*yc 
    #so we solve for xc and yc
    
    A = 1-afCoeff[0, 0]
    B = afCoeff[0, 1]

    xd = afCoeff[0, 2]
    yd = afCoeff[1, 2]

    yc = (yd-B/A*xd)/(B**2/A+A)
    xc = (xd+B*yc)/A

    return xc, yc

def f(c, x, y):
    """
    calculate the algebraic distance between the data points
    and the mean circle centered at c=(xc, yc)
    """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def least_squares_circle(x, y):
    """
    Circle fit using least-squares solver.
    Inputs:

        - coords, list or numpy array with len>2 of the form:
        [
    [x_coord, y_coord],
    ...,
    [x_coord, y_coord]
    ]

    Outputs:

        - xc : x-coordinate of solution center (float)
        - yc : y-coordinate of solution center (float)
        - R : Radius of solution (float)
        - residu : MSE of solution against training data (float)
    """

    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x, y))
    xc, yc = center
    Ri = calc_R(x, y, *center)
    R = Ri.mean()
    residu = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def calc_R(x, y, xc, yc):
    
    """
    calculate the distance of each 2D points from the center (xc, yc)
    """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def initialBoresight(db, frameIDs):

    """ 
    An initial approximation of the boresight from a sequence of spot measurements 

    We calculate the mean position of the spots for each frame, and then fit a circle
    to the sequence of means. As the rotation centre is slightly offset from the image centre,
    the mean positions will trace a circle around the centre of rotation.

    """
    
    xC = []
    yC = []
    #for each frame, pull the spots from database, and calculate mean position
    for frameId in frameIDs:
        # retrieve all spots
        points = loadCentroidsFromDB(db, frameId)

        # get means
        xC.append(np.nanmean(points[:, 1]))
        yC.append(np.nanmean(points[:, 2]))

    # do the fit
    xCentre, yCentre, radius, residuals = least_squares_circle(xC, yC)
    
    return [xCentre, yCentre]

def refineBoresight(db, frameId1, frameId2, boresightEstimate):
    """
    Refine the boresight measurement to subpixel accuracy.

    - take two frames at different angles, 
    - rotate using the estimated centre and angle,
    - do nearest neighbour matching
    - calculate the affine transform
    - extract the centre of rotation

    Input: 
    frameId1, frameId2: mcs_frame_id for the two sets of points
    boresightEstimate: [xC,yC] returned by initialBoresight
    
    """

    # retrieve two sets of points
    points = loadCentroidsFromDB(db, frameId1)
    points1 = loadCentroidsFromDB(db, frameId2)

    points=points[~np.isnan(points).any(axis=1)]
    points1=points1[~np.isnan(points1).any(axis=1)]
    
    # and their instrumetn rotation
    zenithAngle1, insRot1 = loadTelescopeParametersFromDB(db, frameId1)
    zenithAngle2, insRot2 = loadTelescopeParametersFromDB(db, frameId2)

    thetaDiff = (insRot1 - insRot2) * np.pi/180

    x1 = points[:, 1]
    y1 = points[:, 2]

    # the estimated centre
    x0 = boresightEstimate[0]
    y0 = boresightEstimate[1]

    # rotate to match the second set of point, using theta differnce
    
    x2 = (x1-x0) * np.cos(thetaDiff) - (y1-y0) * np.sin(thetaDiff) + x0
    y2 = (x1-x0) * np.sin(thetaDiff) + (y1-y0) * np.cos(thetaDiff) + y0

    # some bookkeeping for the format expected by nearestNeighbourMatching

    #three sets of poitns, rotated, unrotated and target

    unRot = points[:, 0:3]
    source = np.array([x2, x2, y2]).T
    target = points1[:, 0:3]

    # do nearest neighbour matching on *transformed* values,
    # and return the *untransformed* values matched to the first set

    matchPoint = nearestNeighbourMatchingBore(source, target, unRot)
 
    #gethe affine transform
    afCoeff, xd, yd, sx, sy, rotation = calcAffineTransform(target[:, 0:3], matchPoint[:, 0:3])

    #and extract the centre of rotation from the matrix
    xc, yc = extractRotCent(afCoeff)
        
    return [xc, yc]


def calcBoresight(cmd, db, frameIds, pfsVisitId, writeToDB=True):

    """
    wrapper for boresight calculationg.

    Input: 
       db: database connection
       frameIds: list of mcs_frame_ids for the data set

    Output: 
       returns boresight
       write updated value to database

    """

    boresightEstimate = initialBoresight(db, frameIds)
    boresight = refineBoresight(db, frameIds[0], frameIds[1], boresightEstimate)
    
    cmd.inform(f'text="Boresight values {boresight}."')
    
    if writeToDB is True:
        writeBoresightToDB(db, pfsVisitId, boresight)
    else:
        cmd.inform(f'text="Not writing to DB."')

    return boresight

def loadCentroidsFromDB(db, mcsFrameId):
    """ retrieve a set of centroids from database and return as a numpy array"""
    
    sql = f'select mcs_data.spot_id, mcs_data.mcs_center_x_pix, mcs_data.mcs_center_y_pix from mcs_data where mcs_data.mcs_frame_id={mcsFrameId}'
    df = db.query_dataframe(sql)
    return df.to_numpy()

def loadTelescopeParametersFromDB(db, frameId):

    sql = f'SELECT mcs_exposure.insrot,mcs_exposure.altitude FROM mcs_exposure WHERE mcs_exposure.mcs_frame_id={frameId}'
    df = db.query_dataframe(sql)

    if df['altitude'][0] < -99:
        zenithAngle = 90
    else:
        zenithAngle = 90-df['altitude'][0]

    insRot = df['insrot'][0]

    return zenithAngle, insRot

def writeBoresightToDB(db, pfsVisitId, boresight):
    """ write boresight to database with current timestamp """
    
    dt = datetime.now(timezone.utc)
   
    df = pd.DataFrame({'pfs_visit_id': [pfsVisitId], 'mcs_boresight_x_pix': [boresight[0]], 'mcs_boresight_y_pix': [boresight[1]],
                       'calculated_at': [dt]})

    db.insert('mcs_boresight', df)

def calcAffineTransform(pos1, pos2):
    """

    given two sets of registered points, estimate the rigid transformation
    this is a wrapper for the cv2 routine
    Returns transformation matrix and extracted parameters (rotation, translation, scale)

    input:
    pos1 input positions in nx3 shape, first column is an id number, next two coordinates
    xx, yy: transformed positions
    getVales: if == 1, return parameters too

    output: 
    transformation: matrix 
    xd, yd: translations
    sx, sy: scalings
    rotation: rotation (radians)

    """

    sz = pos1.shape[0]
    # turn data into right form
    pts1 = np.zeros((1, sz, 2))
    pts2 = np.zeros((1, sz, 2))

    pts1[0, :, 0] = pos1[:, 1]
    pts1[0, :, 1] = pos1[:, 2]

    pts2[0, :, 0] = pos2[:, 1]
    pts2[0, :, 1] = pos2[:, 2]

    #float32 is needed
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # calculate the transformation
    #transformation = cv2.estimateRigidTransform(pts1, pts2, False)

    afCoeff, inlier = cv2.estimateAffinePartial2D(pts1, pts2)

    # extract the parameters

    sx = np.sqrt(afCoeff[0, 0]**2+afCoeff[0, 1]**2)
    sy = np.sqrt(afCoeff[1, 0]**2+afCoeff[1, 1]**2)

    xd = afCoeff[0, 2]
    yd = afCoeff[1, 2]

    rotation = np.arctan2(afCoeff[1, 0]/np.sqrt(afCoeff[0, 0]**2+afCoeff[0, 1]**2), 
                          afCoeff[1, 1]/np.sqrt(afCoeff[1, 0]**2+afCoeff[1, 1]**2))

    return afCoeff, xd, yd, sx, sy, rotation

