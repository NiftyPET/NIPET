from pathlib import Path

import cuvec as cu
import h5py
import numpy as np
import pydicom as dcm
from tqdm.auto import trange

from niftypet import nimpa, nipet
from niftypet.nipet.nrm import nrm1


def main():
    mfldr = Path(__file__).parent.parent.parent.parent / 'data_DPUK_raw' # INPUT FOLDER

    rawdtypes = [5, 4] # GE raw data types (we are interested in two only)

    rawdcms = nimpa.dcmsort(mfldr) # identify all DICOM files of the given study
    if len(rawdcms) > 1:
        raise ValueError("More than one acquisition present for raw data")

    opth = mfldr / 'output_raw'

    # output dictionary of raw HDF5 files (norm)
    rawh5 = {}
    for rd in next(iter(rawdcms.values()))['files']:
        # print(rd)
        dhdr = dcm.dcmread(rd)
        dtype = dhdr[0x021, 0x1001].value
        if dtype in rawdtypes and [0x023, 0x1002] in dhdr:
            nimpa.create_dir(opth)
            fout = opth / f"{rd.name.split('.')[0]}_raw-{dtype}.h5"
            fout.write_bytes(dhdr[0x023, 0x1002].value)
            rawh5[f'r{dtype}'] = {'fh5': fout}

    # Normalisation components
    nkys = {
        'sclfcr': 'ScaleFactors/Segment4/scaleFactors',
        'nrm3d': 'SegmentData/Segment4/3D_Norm_Correction',
        'dtpu3d': 'DeadtimeCoefficients/dt_3dcrystalpileUp_factors',
        'ceff': '3DCrystalEfficiency/crystalEfficiency'}             # norm keys
    for r in rawh5:
        f = h5py.File(rawh5[r]['fh5'], 'r')
        for nk, nv in nkys.items():
            if nk == 'nrm3d' and f'{nv}/slice1' in f:
                tmp = np.array(f[f'{nv}/slice1'])
                nrm3d = np.zeros((len(f[nv]),) + tmp.shape, dtype=np.float32)

                for si in f[nv]:
                    ind = int(si[5:])
                    # print('geo norm slice #:', ind)
                    nrm3d[ind - 1] = np.array(f[f'{nv}/{si}'])
                    assert nrm3d[ind - 1].any()

                rawh5[r]['nrm3d'] = nrm3d
                assert nrm3d.any()
            elif nv in f:
                rawh5[r][nk] = np.array(f[nv])

    # check for inconsistencies
    for nk in rawh5['r4']:
        if nk in rawh5['r5'] and nk != 'fh5':
            diff = np.sum(rawh5['r5'][nk] - rawh5['r4'][nk])
            if diff > 1e-5:
                raise ValueError(
                    "Difference in normalisation components in the two HDF5 norm files"
                    f" ({nk}: {diff})")

    Cnt, txLUT, axLUT = nipet.sigaux.init_sig(rawh5['r4']['fh5'])

    sino = np.zeros((Cnt['NAW'],), dtype=np.float32) # TRANSIAXIAL LOR SIGN
    im = np.zeros((Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)

    tv = np.zeros(Cnt['NTV'] * Cnt['NAW'], dtype=np.uint8)
    tt = np.zeros(Cnt['NTT'] * Cnt['NAW'], dtype=np.float32)

    nipet.prjsig.tprj(sino, im, tv, tt, txLUT, Cnt)
    tt.shape = (Cnt['NAW'], Cnt['NTT'])
    tt_ssgn = tt[:, 4]

    # import matplotlib.pyplot as plt
    # plt.matshow(np.reshape(tt_ssgn, (Cnt['NSBINS'], Cnt['NSANGLES'])).T)

    # repeat & reshape the geometric 16 slices corresponding to the detector unit width
    nrm3d = rawh5['r4']['nrm3d'] * rawh5['r4']['sclfcr'].reshape(-1, 1, 1)
    assert nrm3d.any()

    # geometric norm sinogram
    geosn = np.tile(nrm3d, (Cnt['NSANGLES'] // nrm3d.shape[0], 1, 1)).transpose(1, 0, 2)
    assert geosn.any()

    # crystal efficiencies
    ceff = rawh5['r4']['ceff'][:, :Cnt['NCRS']]

    # GENERATE THE NORM SINOGRAM
    # init norm sino
    nrmsn = np.zeros((Cnt['NSN'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
    # temporary pre-allocate sino of crystal efficiencies represented as a vector
    effsn = np.zeros(Cnt['NAW'], dtype=np.float32)

    # PRECALC

    # tt_ssgn_thresh = (tt_ssgn > 0.1).astype(np.uint8)
    # txLUT_c2s = cu.asarray(txLUT['c2s'], np.int32)
    # assert all(txLUT_c2s[c0, c1] == bidx
    #            for c1 in range(Cnt['NCRS']) for c0 in range(Cnt['NCRS'])
    #            if (bidx := txLUT_c2s[c1, c0]) >= 0), "not symmetric"
    # assert all(ceff.shape[1] == i for i in txLUT_c2s.shape)
    txLUT_s2c = cu.asarray(txLUT['s2c'], np.int32)
    assert len({tuple(sorted(c)) for c in txLUT_s2c}) == len(txLUT_s2c), "duplicates in s2c"
    ceff = cu.asarray(ceff)
    tt_ssgn_thresh = cu.asarray((tt_ssgn > 0.1), np.uint8)

    # bidx: transaxial bin indices
    # txLUT_c2s = txLUT['c2s'].astype(np.int64) # int64 is quicker in Python
    # c_bidx = tuple((c1, c0, bidx) for c1 in range(Cnt['NCRS']) for c0 in range(Cnt['NCRS'] - c1)
    #                if (bidx := txLUT_c2s[c1, c0]) >= 0)
    # assert len({i[2] for i in c_bidx}) == len(c_bidx), (
    #   f"{len(c_bidx) - len({i[2] for i in c_bidx})}/{len(c_bidx)}"
    #   " bidx duplicates found")
    # c_min, c_max = Cnt['NCRS'] // 2, 3 * Cnt['NCRS'] // 2
    for li in (pbar := trange(min(Cnt['NRNG'] * 2, len(axLUT['li2sn'])))):
        sni = axLUT['li2sn'][li, 0] # sino index running linearly

        # ring indices
        r0 = axLUT['li2rno'][li, 0]
        r1 = axLUT['li2rno'][li, 1]

        # print(f'+> R0={r0}, R1={r1}, sni={sni}')
        pbar.set_postfix(R0=r0, R1=r1, sni=sni)
        effsn = nrm1(effsn, ceff, r0, r1, txLUT_s2c, tt_ssgn_thresh, dev_id=0)
        assert effsn.any()
        nrmsn[sni] = geosn[sni] * np.reshape(effsn, (Cnt['NSBINS'], Cnt['NSANGLES'])).T

        # if direct sinogram, then skip the next step
        if li < Cnt['NRNG']: continue

        # sinogram & ring indices
        sni = axLUT['li2sn'][li, 1]
        # r0 = axLUT['li2rno'][li, 1]
        # r1 = axLUT['li2rno'][li, 0]
        r0, r1 = r1, r0

        # print(f'-> R0={r0}, R1={r1}, sni={sni}')
        pbar.set_postfix(R0=r0, R1=r1, sni=sni)
        effsn = nrm1(effsn, ceff, r0, r1, txLUT_s2c, tt_ssgn_thresh, dev_id=0)
        assert effsn.any()

        if sni not in range(1, Cnt['NSEG0'], 2):
            nrmsn[sni] = geosn[sni] * np.reshape(effsn, (Cnt['NSBINS'], Cnt['NSANGLES'])).T
        else:
            nrmsn[sni] = 0.25 * (nrmsn[sni] +
                                 (geosn[sni] * np.reshape(effsn,
                                                          (Cnt['NSBINS'], Cnt['NSANGLES'])).T))
        assert effsn.any()
        assert nrmsn[sni].any()
    pbar.close()
    nrm_ = nrmsn.transpose(1, 0, 2)
    # si=23
    # plt.matshow(100*(nrm_[:,si,:]-nrm[:,si,:])/nrm[:,si,:])
    # plt.colorbar()

    if (fnrm := opth / f'nrm_{len(pbar)}.npz').exists():

        def check_diff(x, y):
            mae = np.abs(x - y).mean()
            if mae > 1e-7:
                nrmse = (((x - y)**2).sum() / (y**2).sum())**0.5
                raise ValueError(f"NRMSE:{nrmse}, MAE:{mae}")

        check_diff(np.load(fnrm)["nrm"], nrm_)
    else:
        np.savez_compressed(fnrm, nrm=nrm_)


if __name__ == "__main__":
    try:
        main = profile(main)
    except NameError:
        pass
    main()
