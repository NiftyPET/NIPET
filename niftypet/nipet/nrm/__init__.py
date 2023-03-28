__all__ = ['ge']
import cuvec as cu
import numpy as np

from . import cu_nrm


def ge(effsn, ceff, r0: int, r1: int, txLUT_s2c, tt_ssgn_thresh, dev_id=None, sync=True):
    """GE normalisation helper."""
    if dev_id is False:
        # CPU-only version
        effsn[:] = 0
        for bidx, (c0, c1) in enumerate(txLUT_s2c):
            # bidx: transaxial bin indices, (c0, c1): crystal pair
            effsn[bidx] = ceff[r0, c0] * ceff[r1, c1] if tt_ssgn_thresh[bidx] else ceff[
                r1, c0] * ceff[r0, c1]
        return effsn
    if dev_id is not None:
        cu.dev_set(dev_id)
    res = cu_nrm.ge(cu.asarray(effsn, dtype=np.float32), cu.asarray(ceff, dtype=np.float32),
                    int(r0), int(r1), cu.asarray(txLUT_s2c, dtype=np.int32),
                    cu.asarray(tt_ssgn_thresh, dtype=np.uint8))
    if sync:
        cu.dev_sync()
        # assert res.any()
    return cu.asarray(res)
