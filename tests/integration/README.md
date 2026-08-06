# fps integration tests

Validating an fps change at four levels, each catching what the one below cannot.  Not
for merging to master: `setup_*.sh` carry absolute paths, and the harness assumes this
machine's data trees.

## Two stacks, not interchangeable

| stack | cobraCharmer | use for | motion tests |
|---|---|---|---|
| `setup_physics.sh` | master + `pfiSimServer` | the sim suite, anything asserting motion | pass |
| `setup_offline.sh` | `alefur/tests_offline` | driving the live actor over the hub | **fail by design** |

`pfiSimServer.py` stubs at the *wire* level: a TCP server on localhost:4001 speaking the
FPGA protocol, so `pfi.py` does genuine encoding and decoding, and the only monkey-patch
is `cc.exposeAndExtractPositions`.  `alefur/tests_offline` stubs at the Python level
(`doConnect=False`, a `NullCam`), which lets the actor start with no hardware but means
nothing moves and no position comes back.  Running the sim suite under it gives three
spurious failures.

Both scripts pin every product to a working copy.  Without that, `pfs_instdata` in
particular resolves to the installed version and the config and calibration silently come
from somewhere else.

## 1. The sim suite

```bash
source setup_physics.sh
python sim_dot_convergence.py          # 12 tests, ~3 min
```

Drives the real `FpsCmd` with a fake actor, cmd and visitor.  Two tests carry most of the
value: `finalizeFiberStatus`, which injects a convergence distribution and re-derives the
expected status independently, and `persistedConvergence`, which is the only test of the
*write* path -- real finalize, real FITS round trip, real opdb ingest.  Everything else
stubs `_finalizeWriteIngestPfsConfig`, so without it the columns are computed and thrown
away.

## 2. The live actor

```bash
source setup_offline.sh
setsid nohup ./start_fps.sh > /tmp/fps.log 2>&1 </dev/null &
timeout 150 bash -c 'until oneCmd.py fps ping 2>&1 | grep -q Present; do sleep 4; done'
oneCmd.py fps loadModel
oneCmd.py iic declareCurrentPfsDesign designId=0x59011d629a2ccdb6
oneCmd.py iic moveToPfsDesign noTweak
oneCmd.py iic moveToHome all maskFile=MOD4_group1
```

This is what catches command plumbing, iic argument routing and version mismatch.  iic
runs the *installed* datamodel while fps runs the branch, so it doubles as a
forward-compatibility test of new files against old readers.

`noTweak` is usually needed: `tweakTargetPosition` also tweaks `guideStars`, and an empty
guideStars array gives a SkyCoord with no differentials, so `apply_space_motion` raises
for any design without them.

Needs the mcsActor `alefur/tests_offline` branch to close the loop -- it rewrites replayed
`cobra_match` rows as a measurement of the `cobra_target` fps actually commanded.  Without
it the actor reads back a recording of some other visit and nothing converges.

## 3. Checking the result

```bash
python checkVisit.py 122981
python checkVisit.py 122979 122981          # exit code 1 if any check fails
```

Asserts what a written pfsConfig must satisfy, and that opdb agrees with it row for row:
the command partition, that no cobra reads NOT_SET, that non-science cobras carry no
validation bit, that GOOD implies an accepted target and -- on the convergence path -- a
distance within the recorded tolerance.  Every check derives from the file alone or from
the file against the database; none of them ask fps what it did, which is the point.

## 4. datamodel and pfs_utils

Their tests need `lsst.utils`, absent from rubin10-ics, so they run on **pfsa01**:

```bash
ssh pfsa01 'source /work/stack/loadLSST.bash && conda activate "$LSST_CONDA_ENV_NAME" \
  && cd ~/devel/datamodel && setup -r . && scons -j 16'
```

`setup -j -r .` skips dependencies and `lsst.utils` goes missing -- use `setup -r .`.
pfs_utils has no `sconsUtils` in its table, so set datamodel up first, then `setup -k -r .`
for pfs_utils and run pytest directly.

## Traps

- **Never `pkill -f "ics.fpsActor.main"`** from a shell whose command line contains that
  string: pgrep matches the shell and kills it.  Use `pgrep -f "[i]cs.fpsActor.main"`, and
  keep the launch string out of the same command.
- `actorConfig` is read at actor **startup**, not per command.  After editing
  pfs_instdata: `oneCmd.py fps reloadConfiguration`.
- Actor logs are under `$ICS_MHS_LOGS_ROOT/actors/{fps,mcs}/`, newest non-`current` file.
- A `cobra_match` insert failure names the real constraint in the psql DETAIL line: the
  foreign keys point at **cobra_target** (visit, iteration, cobra_id) and at **mcs_data**
  (mcs_frame_id, spot_id).  Read it rather than guessing.
- Adding a field to `PfsConfig._fields` breaks every datamodel test that constructs one,
  because `_makeInstance` fills each name via `getattr(self, name)`.  Add it to `setUp`
  and `assertPfsConfig` too.
- astropy truncates long FITS card comments at **write** time, not on assignment, so
  inspecting the comment after setting it reports it fine.
