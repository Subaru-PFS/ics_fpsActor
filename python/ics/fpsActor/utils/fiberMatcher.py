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
        fidMatch["global_id"] = -fidMatch["fiducial_fiber_id"].to_numpy()
        fidMatch["x"] = fidMatch["pfi_center_x_mm"].to_numpy()
        fidMatch["y"] = fidMatch["pfi_center_y_mm"].to_numpy()
        fidMatch["xPrior"] = fidMatch["pfi_center_x_mm"].to_numpy()
        fidMatch["yPrior"] = fidMatch["pfi_center_y_mm"].to_numpy()
        fidMatch["type"] = "fiducial"

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

        cobXY['global_id'] = cobXY.cobraId.to_numpy()
        cobXY["type"] = "cobra"

        targets = read_sql(f"""
            SELECT fiber_id, pfi_nominal_x_mm, pfi_nominal_y_mm
            FROM pfs_design_fiber
            INNER JOIN pfs_config ON pfs_config.pfs_design_id = pfs_design_fiber.pfs_design_id
            WHERE pfs_config.visit0 = {nearId}
        """)

        # Merge into cobXY on fiberId
        cobXY = cobXY.merge(targets, how="left", left_on="fiberId", right_on="fiber_id")

        # Optionally rename for consistency
        cobXY = cobXY.rename(columns={
            "pfi_nominal_x_mm": "xTarget",
            "pfi_nominal_y_mm": "yTarget"
        })

        cobXY["xPrior"] = np.nanmean(cobXY[["x", "xTarget"]].to_numpy(), axis=1)
        cobXY["yPrior"] = np.nanmean(cobXY[["y", "yTarget"]].to_numpy(), axis=1)

        return fidMatch, cobXY

    def match(self, visit, iteration=0, searchRadius=1.5):
        """
        Match MCS spots to fibers for a given visit and iteration.
        If searchRadius is set, it overrides the fiber's default armLength for tighter matching (useful for iteration > 0).
        """
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
        ref = pd.concat([fidXY, cobXY], ignore_index=True)

        # MCS spot positions
        mcsPositions = mcsData[["pfi_center_x_mm", "pfi_center_y_mm"]].to_numpy()
        tree = cKDTree(mcsPositions)

        # Matching loop
        matches = []
        for _, fiber in ref.iterrows():
            # Determine search radius
            radius = searchRadius if iteration > 0 else fiber.armLength * 1.05

            # Center for search: use previous position if searchRadius is tight
            center_x = fiber.xPrior if iteration > 0 else fiber.x
            center_y = fiber.yPrior if iteration > 0 else fiber.y

            if any(np.isnan([center_x, center_y])):
                continue

            # Find all spots within the radius
            candidates = tree.query_ball_point([center_x, center_y], r=radius)

            for spot_idx in candidates:
                spot = mcsData.iloc[spot_idx]
                d = np.hypot(spot.pfi_center_x_mm - fiber.x, spot.pfi_center_y_mm - fiber.y)
                matches.append({
                    "spot_id": spot.spot_id,
                    "global_id": fiber.global_id,
                    "fiber_type": fiber.type,
                    "distance_mm": d,
                    "spot_x": spot.pfi_center_x_mm,
                    "spot_y": spot.pfi_center_y_mm,
                    "xyPrior_x": fiber.xPrior,
                    "xyPrior_y": fiber.yPrior
                })

        matchDf = pd.DataFrame(matches)

        if matchDf.empty:
            return pd.DataFrame()  # early return if no matches

        # Compute distance to prior position (if any)
        matchDf["dist_to_prior"] = np.hypot(matchDf["spot_x"] - matchDf["xyPrior_x"],
                                            matchDf["spot_y"] - matchDf["xyPrior_y"])

        # If multiple matches for a fiber, keep the closest to prior (or patrol center for iteration 0)
        matchDf = matchDf.sort_values("dist_to_prior").drop_duplicates("global_id", keep="first")

        # Final merge to get full spot info
        merged = matchDf.merge(mcsData, on="spot_id", how="left")
        return merged

    def cobraMatch(self, visit, iteration=0):
        """Return cobra-only match table with correct columns, padding, and dtypes (like getCobraMatchData)."""
        matches = self.match(visit, iteration)
        cobraMatch = matches[matches.fiber_type == "cobra"].copy()

        # Add cobra_id
        cobraMatch["cobra_id"] = cobraMatch["global_id"]

        # Pad missing cobra_ids
        cobraMatch = cobraMatch.set_index("cobra_id").reindex(np.arange(1, 2395)).reset_index()

        # Replace NaN spot_id with -1 and ensure int
        cobraMatch["spot_id"] = cobraMatch["spot_id"].fillna(-1).astype(int)

        # adding visit and iteration
        cobraMatch["pfs_visit_id"] = visit
        cobraMatch["iteration"] = iteration

        # Columns and dtypes to match the expected format
        columns_and_types = {
            "pfs_visit_id": "int64",
            "iteration": "int64",
            "cobra_id": "int64",
            "spot_id": "int64",
            "pfi_center_x_mm": "float64",
            "pfi_center_y_mm": "float64",
            "mcs_center_x_pix": "float64",
            "mcs_center_y_pix": "float64"
        }

        # Ensure all required columns are present with correct type
        for col, dtype in columns_and_types.items():
            if col not in cobraMatch.columns:
                cobraMatch[col] = np.nan if "float" in dtype else -1
            cobraMatch[col] = cobraMatch[col].astype(dtype)

        # Return ordered DataFrame
        cobraMatch = cobraMatch[list(columns_and_types.keys())]

        # updating prior
        updatePrior = cobraMatch[cobraMatch.cobra_id.isin(self.cobXY.cobraId.to_numpy())].sort_values('cobra_id')
        np.testing.assert_equal(updatePrior.cobra_id.to_numpy(), self.cobXY.cobraId.to_numpy())

        # updating prior
        self.cobXY['xPrior'] = updatePrior.pfi_center_x_mm.to_numpy()
        self.cobXY['yPrior'] = updatePrior.pfi_center_y_mm.to_numpy()

        return cobraMatch
