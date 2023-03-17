__all__ = ['nrm1']
import cuvec as cu
import numpy as np

from . import casper_nrm


def nrm1(effsn, ceff, r0: int, r1: int, NCRS: int, txLUT_c2s, tt_ssgn_thresh, dev_id=None,
         sync=True):
    assert all(ceff.shape[1] == i for i in txLUT_c2s.shape)
    if dev_id is False:
        # CPU-only version
        effsn[:] = 0
        for c1 in range(NCRS):
            for c0 in range(NCRS):
                if (bidx := txLUT_c2s[c1, c0]) >= 0:
                    effsn[bidx] = (ceff[r0, c0] *
                                   ceff[r1, c1] if tt_ssgn_thresh[bidx] else ceff[r1, c0] *
                                   ceff[r0, c1])
        return effsn
    if dev_id is not None:
        cu.dev_set(dev_id)
    res = casper_nrm.nrm1(cu.asarray(effsn, dtype=np.float32), cu.asarray(ceff, dtype=np.float32),
                          int(r0), int(r1), int(NCRS), cu.asarray(txLUT_c2s, dtype=np.int32),
                          cu.asarray(tt_ssgn_thresh, dtype=np.uint8))
    if sync:
        cu.dev_sync()
    return cu.asarray(res)
