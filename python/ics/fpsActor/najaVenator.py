import io
from importlib import reload

import numpy as np
import pandas as pd
import psycopg2
from pfs.utils.database import opdb

class NajaVenator(object):
    """ 
        A class of interface providing connection capability with opDB. 
        Naja is the genus name of cobra and venator is latin word for hunter.  

    """

    def __init__(self):
        self._dbConn = opdb.OpDB()

    @staticmethod
    def connect():
        """Return connection object, password needs to be defined in /home/user/.pgpass."""
        return psycopg2.connect(host='db-ics', dbname='opdb', user='pfs')

    def reform(self, df, nameMap=None, typeMap=None):
        if nameMap is not None:
            colNames = nameMap.keys()
            df = df[colNames].rename(columns=nameMap, inplace=True)

        if typeMap is not None:
            df = df.astype(typeMap)

        return df

    def readFFConfig(self):
        """Read positions of all fidicial fibers."""

        df = self._dbConn.query_dataframe("SELECT fiducial_fiber_id as id, ff_center_on_pfi_x_mm as x, ff_center_on_pfi_y_mm as y "
                                          "FROM fiducial_fiber_geometry WHERE fiducial_fiber_calib_id = 1")
        df = self.reform(typeMap=dict(id='i4',
                                       x='f4',
                                       y='f4'))
        return df

    def readCentroid(self, frameId):
        """ Read centroid information from database. This requires INSTRM-1110."""

        df = self._dbConn.query_dataframe(f"""SELECT * from mcs_data WHERE mcs_frame_id={frameId}""")

        # We got a full table, with original names. Trim and rename to
        # what is expected here.
        renames = dict(mcs_frame_id='mcsId',
                       spot_id='fiberId',
                       mcs_center_x_pix='centroidx',
                       mcs_center_y_pix='centroidy')

        df = self.reform(nameMap=renames,
                         typeMap=dict(centroidx='f4',
                                      centroidy='f4'))
        return df

    def readTelescopeInform(self, frameId):
        raise NotImplementedError()

        buf = io.StringIO()

        cmd = f"""COPY (SELECT * from mcs_exposure WHERE mcs_frame_id={frameId}) to stdout delimiter ','"""

        with self.connect() as conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        buf.seek(0, 0)

        arr = np.genfromtxt(buf, dtype='f4', delimiter=',', usecols=range(7))
        arr = arr[[0, 3, 4, 5]].reshape(1, 4)

        return pd.DataFrame(arr, columns=['frameId', 'alt', 'azi', 'instrot'])

    def writeBoresightTable(self, data):
        cmd = f""" INSERT INTO mcs_boresight (pfs_visit_id, mcs_boresight_x_pix, mcs_boresight_y_pix, calculated_at)  
        VALUES ({data['visitid']}, {data['xc']}, {data['yc']}, 'now')"""

        with self.connect() as conn:
            with conn.cursor() as curs:
                curs.execute(cmd)

    def writeCobraConfig(self, matchCatalog, frameid):
        raise NotImplementedError()

        measBuf = io.StringIO()
        new = matchCatalog.dropna(thresh=6)
        new.to_csv(measBuf, sep=',', index=False, header=False)

        # np.savetxt(measBuf, centArr[:,1:7], delimiter=',', fmt='%0.6g')
        measBuf.seek(0, 0)

        colname = ['"fiberId"', '"mcsId"', '"pfiNominal_x"', '"pfiNominal_y"', '"pfiCenter_x"', '"pfiCenter_y"',
                   '"mcsCenter_x"', '"mcsCenter_y"', '"pfiDiff_x"', '"pfiDiff_y"', ]

        buf = io.StringIO()
        for l_i in range(len(new)):
            buf.write(str(measBuf.readline()))

        buf.seek(0, 0)

        with self.connect() as conn:
            with conn.cursor() as curs:
                curs.COPY_from(buf, '"CobraConfig"', ',', columns=colname)

        buf.seek(0, 0)

        return buf

    def writeTelescopeInform(self, data):
        pass


class CobraTargetTable(object):
    def __init__(self, visitid, tries, calibModel, designID, goHome = False):
        self._dbConn = opdb.OpDB()
        self.visitid = visitid
        self.tries = tries
        self.designID = designID
        self.goHome = goHome
        
        if self.goHome is True:
            self.iteration = tries+1     
        else:
            self.iteration = tries

        self.calibModel = calibModel

    def makeTargetTable(self, moves, cobraCoach, goodIdx):
        """Make the target table for the convergence move."""
        cc = cobraCoach

        pfs_config_id = self.designID

        firstStepMove = moves['position'][:, 0]
        firstThetaAngle = moves['thetaAngle'][:, 0]
        firstPhiAngle = moves['phiAngle'][:, 0]

        targetStepMove = moves['position'][:, 2]
        targetThetaAngle = moves['thetaAngle'][:, 2]
        targetPhiAngle = moves['phiAngle'][:, 2]

        targetTable = {'pfs_visit_id': [],
                       'iteration': [],
                       'cobra_id': [],
                       'pfs_config_id': [],
                       'pfi_nominal_x_mm': [],
                       'pfi_nominal_y_mm': [],
                       'pfi_target_x_mm': [],
                       'pfi_target_y_mm': [],
                       'flags':[]
                       }

        for iteration in range(self.iteration):
            for idx in range(cc.nCobras):
                targetTable['pfs_visit_id'].append(self.visitid)
                targetTable['pfs_config_id'].append(pfs_config_id)

                targetTable['cobra_id'].append(idx + 1)
                targetTable['iteration'].append(iteration)

                targetTable['pfi_nominal_x_mm'].append(self.calibModel.centers[idx].real)
                targetTable['pfi_nominal_y_mm'].append(self.calibModel.centers[idx].imag)
                targetTable['flags'].append(0)

                if idx in cc.badIdx or idx not in goodIdx:
                    # Using cobra center for bad cobra targets
                    targetTable['pfi_target_x_mm'].append(self.calibModel.centers[idx].real)
                    targetTable['pfi_target_y_mm'].append(self.calibModel.centers[idx].imag)
                else:
                    if self.goHome is True:
                        if iteration == 0:
                            targetTable['pfi_target_x_mm'].append(self.calibModel.centers[idx].real)
                            targetTable['pfi_target_y_mm'].append(self.calibModel.centers[idx].imag)
                        elif iteration == 1 or iteration == 2:
                            targetTable['pfi_target_x_mm'].append(firstStepMove[goodIdx == idx].real[0])
                            targetTable['pfi_target_y_mm'].append(firstStepMove[goodIdx == idx].imag[0])
                        else:
                            targetTable['pfi_target_x_mm'].append(targetStepMove[goodIdx == idx].real[0])
                            targetTable['pfi_target_y_mm'].append(targetStepMove[goodIdx == idx].imag[0])
                    else:
                    
                        if iteration < 2:
                            targetTable['pfi_target_x_mm'].append(firstStepMove[goodIdx == idx].real[0])
                            targetTable['pfi_target_y_mm'].append(firstStepMove[goodIdx == idx].imag[0])
                        else:
                            targetTable['pfi_target_x_mm'].append(targetStepMove[goodIdx == idx].real[0])
                            targetTable['pfi_target_y_mm'].append(targetStepMove[goodIdx == idx].imag[0])

        self.dataTable = pd.DataFrame(targetTable)

        return self.dataTable

    def writeTargetTable(self):
        """Write self.dataTable to cobra_target table."""

        self._dbConn.insert_dataframe("cobra_target", self.dataTable)
