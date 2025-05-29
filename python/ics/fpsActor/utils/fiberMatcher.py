import numpy as np
import pandas as pd
from ics.fpsActor.utils.alfUtils import sgfm, read_sql, getMcsDataOnPfi
from scipy.spatial import cKDTree


class FiberMatcher:
    """Matches MCS spots to fiducials and cobras using spatial proximity and arm-length constraints."""

    def __init__(self, nearConvergenceId, notMovingRadius=1.0):
        """Initialize with a reference visit ID and assign arm lengths to static fibers."""
        self.nearConvergenceId = nearConvergenceId
        self.notMovingRadius = notMovingRadius
        self.fidXY, self.cobXY = self._getBasePosition()

    def _toFwhmPix(self, df):
        """Compute FWHM (in pixels) from second moments assuming a Gaussian profile."""
        mxx = df["mcs_second_moment_x_pix"]
        myy = df["mcs_second_moment_y_pix"]
        mxy = df["mcs_second_moment_xy_pix"]
        rms = np.sqrt(np.sqrt(mxx * myy - mxy ** 2))
        return 2 * np.sqrt(2 * np.log(2)) * rms

    def _getBasePosition(self):
        """Query fiducial and cobra positions from past visits and compute base positions."""
        nearId = self.nearConvergenceId
        r = self.notMovingRadius

        # 1. Fiducial match data
        fidQuery = f"""
            SELECT ffm.pfs_visit_id,
                   ffm.fiducial_fiber_id,
                   ffm.pfi_center_x_mm, ffm.pfi_center_y_mm,
                   md.mcs_center_x_pix, md.mcs_center_y_pix,
                   md.mcs_second_moment_x_pix, md.mcs_second_moment_y_pix,
                   md.mcs_second_moment_xy_pix, md.peakvalue
            FROM fiducial_fiber_match ffm
            INNER JOIN mcs_data md
                ON md.mcs_frame_id = ffm.pfs_visit_id * 100
                AND md.spot_id = ffm.spot_id
            WHERE ffm.pfs_visit_id IN ({nearId}, {nearId - 2})
              AND ffm.iteration = 0
        """
        fidMatch = read_sql(fidQuery)
        fidMatch["fwhm_pix"] = self._toFwhmPix(fidMatch)
        fidAtHome = fidMatch[fidMatch.pfs_visit_id == nearId - 2]
        fidMatch = fidMatch.groupby("fiducial_fiber_id")[["pfi_center_x_mm", "pfi_center_y_mm"]].mean()
        fidMatch["peakvalue"] = fidAtHome["peakvalue"].to_numpy()
        fidMatch["fwhm_pix"] = fidAtHome["fwhm_pix"].to_numpy()
        fidMatch["armLength"] = r
        fidMatch = fidMatch.reset_index()

        # 2. Cobra match data
        cobQuery = f"""
            SELECT cm.pfs_visit_id, cm.iteration, cm.cobra_id,
                   cm.pfi_center_x_mm, cm.pfi_center_y_mm,
                   ct.pfi_target_x_mm, ct.pfi_target_y_mm,
                   md.mcs_center_x_pix, md.mcs_center_y_pix,
                   md.mcs_second_moment_x_pix, md.mcs_second_moment_y_pix,
                   md.mcs_second_moment_xy_pix, md.peakvalue
            FROM cobra_match cm
            INNER JOIN cobra_target ct
                ON ct.pfs_visit_id = cm.pfs_visit_id
               AND ct.iteration = cm.iteration
               AND ct.cobra_id = cm.cobra_id
            INNER JOIN mcs_data md
                ON md.mcs_frame_id = cm.pfs_visit_id * 100 + cm.iteration
               AND md.spot_id = cm.spot_id
            WHERE cm.pfs_visit_id IN ({nearId}, {nearId - 2})
              AND cm.iteration = 0
        """
        cobMatch = read_sql(cobQuery)
        cobMatch["fwhm_pix"] = self._toFwhmPix(cobMatch)
        atHome = cobMatch[cobMatch.pfs_visit_id == nearId - 2]
        cobraPos = cobMatch.groupby("cobra_id")[["pfi_center_x_mm", "pfi_center_y_mm"]].mean()

        cobXY = sgfm.copy()
        cobXY["x"] = cobraPos["pfi_center_x_mm"].to_numpy()
        cobXY["y"] = cobraPos["pfi_center_y_mm"].to_numpy()
        cobXY["peakvalue"] = atHome["peakvalue"].to_numpy()
        cobXY["fwhm_pix"] = atHome["fwhm_pix"].to_numpy()
        cobXY = cobXY[~cobXY.FIBER_BROKEN_MASK]
        cobXY.loc[~cobXY.COBRA_OK_MASK, "armLength"] = r

        return fidMatch, cobXY

    def match(self, visit, iteration=0):
        # Load MCS data
        mcsData = getMcsDataOnPfi(visit, iteration)
        mcsData = mcsData[[
            "mcs_frame_id", "spot_id", "mcs_center_x_pix", "mcs_center_y_pix",
            "mcs_second_moment_x_pix", "mcs_second_moment_y_pix", "mcs_second_moment_xy_pix",
            "bgvalue", "peakvalue", "pfi_center_x_mm", "pfi_center_y_mm"
        ]].dropna()

        # Prepare reference catalog
        fidXY = self.fidXY.copy()
        cobXY = self.cobXY.copy()

        fidXY["fiber_id"] = -fidXY["fiducial_fiber_id"].to_numpy()
        fidXY["x"] = fidXY["pfi_center_x_mm"].to_numpy()
        fidXY["y"] = fidXY["pfi_center_y_mm"].to_numpy()
        fidXY["type"] = "fiducial"

        cobXY = cobXY.rename(columns={"cobraId": "fiber_id"})
        cobXY["type"] = "cobra"

        ref = pd.concat([fidXY, cobXY], ignore_index=True)
        refPositions = np.column_stack((ref["x"], ref["y"]))
        tree = cKDTree(refPositions)

        # Match
        mcsPositions = mcsData[["pfi_center_x_mm", "pfi_center_y_mm"]].to_numpy()
        dist, idx = tree.query(mcsPositions, distance_upper_bound=6.0)

        matches = []
        for i, (d, j) in enumerate(zip(dist, idx)):
            if j >= len(ref):
                continue
            if d <= ref.iloc[j].armLength:
                matches.append({
                    "spot_id": mcsData.iloc[i].spot_id,
                    "fiber_id": ref.iloc[j].fiber_id,
                    "fiber_type": ref.iloc[j].type,
                    "distance_mm": d
                })

        matches = pd.DataFrame(matches)
        merged = matches.merge(mcsData, on="spot_id", how="left")
        return merged
