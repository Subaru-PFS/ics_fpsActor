"""Assert the invariants a written pfsConfig must satisfy, and that opdb agrees with it.

The sim suite covers the code paths; this covers the product.  Run it after driving the
actor -- `iic moveToPfsDesign`, `iic moveToHome`, a dot design -- and it decides whether
the visit is correct rather than leaving someone to read the numbers and judge.

Every check is derived from the file alone, or from the file against the database.  None
of them ask fps what it did, which is the point: if a reader needs fps to interpret a
pfsConfig, the columns are not carrying what they claim to.

    python checkVisit.py 122981
    python checkVisit.py 122981 --dirName /home/alefur/data/raw/2026-08-06/pfsConfig
"""

import argparse
import glob
import os

import numpy as np
from pfs.datamodel import (PfsConfig, TargetType, FiberStatus, CobraCommand,
                           TargetValidation)
from pfs.utils.database.opdb import OpDB

N_COBRAS = 2394

SCIENCE_TARGET_TYPES = [TargetType.SCIENCE, TargetType.SKY, TargetType.FLUXSTD]

VALIDATION_REASONS = int(TargetValidation.NOT_FINITE
                         | TargetValidation.TOO_CLOSE_TO_CENTER
                         | TargetValidation.TOO_FAR_FROM_CENTER
                         | TargetValidation.FIDUCIAL_INTERFERENCE)


class Result:
    """Outcome of one visit's checks, accumulated so every failure is reported at once.

    A checker that stops at the first failure hides the others, and the others are
    usually what explain it.
    """

    def __init__(self, visit):
        self.visit = visit
        self.checks = []

    def record(self, ok, name, detail=""):
        self.checks.append((bool(ok), name, detail))
        return ok

    @property
    def failed(self):
        return [check for check in self.checks if not check[0]]

    def report(self):
        """Print every check, and return True if all passed."""
        for ok, name, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))
        print(f"\n  {len(self.checks) - len(self.failed)}/{len(self.checks)} passed"
              f" for visit {self.visit}")
        return not self.failed


def findPfsConfig(visit, dirName=None):
    """Locate and read the pfsConfig for a visit.

    Parameters
    ----------
    visit : `int`
    dirName : `str`, optional
        Directory holding it; searched under the raw tree when not given.

    Returns
    -------
    `pfs.datamodel.PfsConfig`
    """
    patterns = ([os.path.join(dirName, f"pfsConfig-0x*-{visit:06d}.fits")] if dirName else
                [f"/home/alefur/data/raw/*/pfsConfig/pfsConfig-0x*-{visit:06d}.fits",
                 f"/data/raw/*/pfsConfig/pfsConfig-0x*-{visit:06d}.fits"])
    for pattern in patterns:
        found = glob.glob(pattern)
        if found:
            path = sorted(found)[-1]
            designId = int(os.path.basename(path).split("-")[1], 16)
            return PfsConfig.read(designId, visit, dirName=os.path.dirname(path))

    raise FileNotFoundError(f"no pfsConfig for visit {visit} in {patterns}")


def checkCobraCommand(cfg, result):
    """cobraCommand must partition the cobras and agree with what else the file says."""
    onCobra = cfg.cobraConfig()
    command = onCobra.cobraCommand
    counts = {CobraCommand(value).name: int(n)
              for value, n in zip(*np.unique(command, return_counts=True))}

    result.record(len(onCobra) == N_COBRAS, "cobraConfig returns every cobra",
                  f"{len(onCobra)} rows")
    result.record(np.array_equal(onCobra.cobraId, np.arange(1, N_COBRAS + 1)),
                  "cobraConfig is sorted by cobraId")
    result.record(sum(counts.values()) == N_COBRAS, "cobraCommand partitions the fleet",
                  str(counts))
    result.record(not (command == int(CobraCommand.NOT_SET)).any(),
                  "no cobra reads NOT_SET",
                  "NOT_SET is only for a file predating the column")

    # A cobra driven to a black dot is one the design parked or fps refused; a cobra
    # driven to its design target must have been asked to converge.
    converged = command == int(CobraCommand.CONVERGE)
    isScience = np.isin(onCobra.targetType, SCIENCE_TARGET_TYPES)
    result.record(bool(np.all(isScience[converged])),
                  "only science targets were converged",
                  f"{int((converged & ~isScience).sum())} non-science converged")

    return onCobra, counts


def checkValidationMask(onCobra, result):
    """The mask judges a target, so it is zero wherever there was none."""
    mask = onCobra.targetValidationMask
    isScience = np.isin(onCobra.targetType, SCIENCE_TARGET_TYPES)

    result.record(not (mask & int(TargetValidation.NOT_SET)).any(),
                  "no cobra reads NOT_SET in the validation mask")
    result.record(bool((mask[~isScience] == 0).all()),
                  "non-science cobras carry no validation bit",
                  f"{int((mask[~isScience] != 0).sum())} do")
    result.record(bool(((mask & ~VALIDATION_REASONS) == 0).all()),
                  "only defined reason bits are set")

    # A refused target is never commanded to pfiNominal, so it cannot be within
    # tolerance: this is the invariant that lets every existing science & GOOD
    # selection keep working untouched.
    refused = (mask & VALIDATION_REASONS) != 0
    good = onCobra.fiberStatus == FiberStatus.GOOD
    result.record(not (refused & good).any(), "GOOD implies an accepted target",
                  f"{int((refused & good).sum())} GOOD cobras carry a reason bit")

    return int(refused.sum())


def checkFiberStatus(cfg, onCobra, result):
    """fiberStatus must be recomputable from the file, and MASKED must be gone."""
    result.record(not (cfg.fiberStatus == FiberStatus.MASKED).any(),
                  "FiberStatus.MASKED is retired")

    tolerance = cfg.convergenceParams.get("distanceTolerance")
    distance = np.hypot(*(onCobra.pfiCenter - onCobra.pfiNominal).T)
    isScience = np.isin(onCobra.targetType, SCIENCE_TARGET_TYPES)
    good = onCobra.fiberStatus == FiberStatus.GOOD

    if tolerance is None:
        # Snapshot: nothing was commanded, so GOOD claims visibility, not accuracy, and
        # BLACKSPOT is the only other outcome a detection test can produce.
        result.record(not (onCobra.fiberStatus == FiberStatus.NOTCONVERGED).any(),
                      "snapshot path writes no NOTCONVERGED")
        return "snapshot"

    # Convergence: GOOD is a positive claim a reader can recompute.
    result.record(bool(np.all(distance[isScience & good] <= tolerance)),
                  f"every science GOOD is within {tolerance} mm",
                  f"max {np.nanmax(distance[isScience & good]):.4f} mm"
                  if (isScience & good).any() else "none")
    notConverged = onCobra.fiberStatus == FiberStatus.NOTCONVERGED
    result.record(bool(np.all(~(distance[isScience & notConverged] <= tolerance))),
                  "every science NOTCONVERGED fails that test")
    result.record(not (onCobra.fiberStatus == FiberStatus.BLACKSPOT).any(),
                  "convergence path writes no BLACKSPOT")

    return "convergence"


def checkAgainstOpdb(cfg, result, db=None):
    """The database must say exactly what the file says."""
    db = OpDB() if db is None else db
    visit = cfg.visit

    fibers = db.query_dataframe(
        f"SELECT fiber_id, fiber_status, target_validation_mask, cobra_command "
        f"FROM pfs_config_fiber WHERE visit0 = {visit} ORDER BY fiber_id")
    if not len(fibers):
        result.record(False, "pfs_config_fiber has rows", "none found")
        return

    order = np.argsort(cfg.fiberId)
    result.record(len(fibers) == len(cfg.fiberId), "opdb has one row per fiber",
                  f"{len(fibers)} vs {len(cfg.fiberId)}")
    for column, values in (("fiber_status", cfg.fiberStatus),
                           ("target_validation_mask", cfg.targetValidationMask),
                           ("cobra_command", cfg.cobraCommand)):
        result.record(np.array_equal(fibers[column].to_numpy(), values[order]),
                      f"opdb {column} matches the file")

    row = db.query_dataframe(
        f"SELECT converg_num_iter, converg_elapsed_time, converg_tolerance, "
        f"converg_distance_threshold, target_fallback_invalid, fiducial_check_skipped "
        f"FROM pfs_config WHERE visit0 = {visit}")
    if not result.record(len(row) == 1, "pfs_config has exactly one row",
                         f"{len(row)} rows"):
        return

    params = dict(cfg.convergenceParams)
    for column, key in (("converg_num_iter", "numIterations"),
                        ("converg_tolerance", "requestedTolerance"),
                        ("converg_distance_threshold", "distanceTolerance"),
                        ("target_fallback_invalid", "targetFallbackInvalid"),
                        ("fiducial_check_skipped", "fiducialCheckSkipped")):
        fromDb, fromFile = row.iloc[0][column], params.get(key)
        if fromFile is None:
            same = fromDb is None or (isinstance(fromDb, float) and np.isnan(fromDb))
        elif isinstance(fromFile, float):
            same = fromDb is not None and abs(float(fromDb) - fromFile) < 1e-6
        else:
            same = fromDb is not None and type(fromFile)(fromDb) == fromFile
        result.record(same, f"opdb {column} matches convergenceParams",
                      f"{fromDb!r} vs {fromFile!r}")


def checkVisit(visit, dirName=None, db=None):
    """Run every check for one visit.

    Parameters
    ----------
    visit : `int`
    dirName : `str`, optional
        Directory holding the pfsConfig.
    db : `pfs.utils.database.opdb.OpDB`, optional

    Returns
    -------
    `Result`
    """
    cfg = findPfsConfig(visit, dirName=dirName)
    result = Result(visit)

    print(f"visit {visit}  {cfg.filename}\n  design: {cfg.designName}")
    onCobra, counts = checkCobraCommand(cfg, result)
    refused = checkValidationMask(onCobra, result)
    path = checkFiberStatus(cfg, onCobra, result)
    checkAgainstOpdb(cfg, result, db=db)

    print(f"\n  path={path}  commands={counts}  refused targets={refused}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visit", type=int, nargs="+", help="visit(s) to check")
    parser.add_argument("--dirName", default=None, help="directory holding the pfsConfig")
    args = parser.parse_args()

    ok = True
    for visit in args.visit:
        ok &= checkVisit(visit, dirName=args.dirName).report()
        print()

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
