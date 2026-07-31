"""motorScales.py — persist the adaptive motor-ontime scaling a convergence learns.

moveToPfsDesign resets the scaling to 1.0 before it moves anything, so every
convergence relearns the correction from scratch and then discards it.  This writes
the learned state out before that happens, so successive convergences can be
compared: a correction that repeats is a per-cobra calibration worth persisting, one
that does not is the loop fitting noise.

The state is cobraState.motorScales, a module-level dict keyed by
(cobraId, motor, direction) with one float per entry.  It is deliberately NOT in the
calibModel -- scaleMotorOntime writes only to the dict, and adjustThetaOnTime reads
it back and multiplies the calibModel's unmodified base on-time -- so the base and
the correction stay separable and the scale means exactly `used / base`.

The base on-times are written alongside the scale.  Without them a scale is not
interpretable later, because the calibModel it was measured against may have been
replaced by the time anyone reads the file.
"""
import numpy as np

from ics.cobraCharmer import cobraState

# calibModel on-time field per (motor, direction, speed).  cw maps to Fwd and ccw to
# Rev; theta is suffix 1 and phi suffix 2.  The scale is held per (cobra, motor,
# direction) and applies to whichever speed the move used, so both are recorded.
_ONTIME_FIELD = {
    ('theta', 'cw', 'slow'): 'motorOntimeSlowFwd1',
    ('theta', 'ccw', 'slow'): 'motorOntimeSlowRev1',
    ('theta', 'cw', 'fast'): 'motorOntimeFwd1',
    ('theta', 'ccw', 'fast'): 'motorOntimeRev1',
    ('phi', 'cw', 'slow'): 'motorOntimeSlowFwd2',
    ('phi', 'ccw', 'slow'): 'motorOntimeSlowRev2',
    ('phi', 'cw', 'fast'): 'motorOntimeFwd2',
    ('phi', 'ccw', 'fast'): 'motorOntimeRev2',
}

FILENAME = 'motor_scales.csv'


def collectMotorScales(cc):
    """Current contents of cobraState.motorScales, with the base on-times.

    Returns
    -------
    rows : list of dict
        One per (cobra, motor, direction) the convergence actually adjusted, sorted.
        A cobra absent from the file was never adjusted, which means a scale of 1.0 --
        that is the dict's own convention and it is preserved rather than padded out,
        so the file says what was learned and not what was merely possible.
    """
    cm = cc.calibModel
    rows = []

    for (cobraId, motor, direction), scale in cobraState.motorScales.items():
        row = dict(cobra_id=int(cobraId) + 1, motor=motor, direction=direction,
                   scale=float(scale))
        for speed in ('slow', 'fast'):
            field = _ONTIME_FIELD.get((motor, direction, speed))
            arr = getattr(cm, field, None) if field else None
            row[f'base_ontime_{speed}'] = (float(arr[cobraId])
                                           if arr is not None and cobraId < len(arr)
                                           else np.nan)
        rows.append(row)

    return sorted(rows, key=lambda r: (r['cobra_id'], r['motor'], r['direction']))


def saveMotorScales(cc, dataPath, cmd=None):
    """Write the learned scaling beside moves.npy.  Never raises.

    Call at the end of a convergence, before the next one resets the state.  A
    failure here must not fail the convergence: this is a diagnostic, and the move
    has already happened by the time it runs.

    Returns
    -------
    nRows : int
        Entries written; 0 if nothing was learned or the write failed.
    """
    try:
        rows = collectMotorScales(cc)
        path = dataPath / FILENAME

        with open(path, 'w') as fh:
            fh.write('cobra_id,motor,direction,scale,base_ontime_slow,base_ontime_fast\n')
            for r in rows:
                fh.write('%d,%s,%s,%.6f,%.6f,%.6f\n'
                         % (r['cobra_id'], r['motor'], r['direction'], r['scale'],
                            r['base_ontime_slow'], r['base_ontime_fast']))

        if cmd is not None and rows:
            s = np.array([r['scale'] for r in rows])
            cmd.inform(f'text="motorScales: {len(rows)} entries written to {FILENAME} '
                       f'(median {np.median(s):.3f}, range {s.min():.3f}-{s.max():.3f})"')
        elif cmd is not None:
            cmd.inform('text="motorScales: nothing learned this convergence"')

        return len(rows)

    except Exception as e:
        if cmd is not None:
            cmd.warn(f'text="motorScales: write failed ({e}); convergence unaffected"')
        return 0
