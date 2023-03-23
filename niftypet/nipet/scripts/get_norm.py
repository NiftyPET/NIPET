import numpy as np
from niftypet import nipet, nimpa

from pathlib import Path
import pydicom as dcm
import h5py

# > GE raw data types (we are interested in two only)
rawdtypes = [5,4]


#======================================================
## SET THE INPUT FOLDER
mfldr = Path('/data/DPUK_raw_2/')

# mfldr = Path('/home/pawel/data/petpp/20170516_e00179_ANON179')
# mfldr = mfldr/'raw'
#======================================================



#-------------------------------------------
# > identify all DICOM files of the given study
rawdcms = nimpa.dcmsort(mfldr)

if len(rawdcms)>1:
    raise ValueError('More than one acquisition present for raw data')
else:
    # > the scan dictionary key
    kscn = next(iter(rawdcms))
#-------------------------------------------

opth = mfldr/'output_raw'

# > output dictionary of raw HDF5 files (norm)
rawh5 = {}
for rd in rawdcms[kscn]['files']:
    #print(rd)
    dhdr = dcm.dcmread(rd)
    dtype = dhdr[0x021,0x1001].value

    if dtype in rawdtypes and [0x023,0x1002] in dhdr:
        nimpa.create_dir(opth)
        fout = opth/(rd.name.split('.')[0]+f'_raw-{dtype}.h5')
        nf = open(fout, 'wb')
        nf.write(dhdr[0x023,0x1002].value)
        rawh5['r'+str(dtype)] = {}
        rawh5['r'+str(dtype)]['fh5'] = fout


#-----------------------------
# > Normalisation components:
#-----------------------------
# norm keys
nkys = dict(
    sclfcr='ScaleFactors/Segment4/scaleFactors',
    nrm3d='SegmentData/Segment4/3D_Norm_Correction',
    dtpu3d='DeadtimeCoefficients/dt_3dcrystalpileUp_factors',
    ceff='3DCrystalEfficiency/crystalEfficiency')

for r in rawh5:
    f = h5py.File(rawh5[r]['fh5'], 'r')

    for nk in nkys:
        if nk=='nrm3d' and nkys[nk]+'/slice1' in f:
            tmp = np.array(f[nkys[nk]+'/slice1'])
            nrm3d = np.zeros((len(f[nkys[nk]]),)+tmp.shape, dtype=np.float32)
            for si in f[nkys[nk]]:
                ind = int(si[5:])
                print('geo norm slice #:', ind)
                nrm3d[ind-1,...] = np.array(f[nkys[nk]+'/'+si])
            rawh5[r][nk] = nrm3d
        elif nkys[nk] in f:
           rawh5[r][nk] = np.array(f[nkys[nk]])


# > check for any inconsistencies
for nk in rawh5['r4']:
    if nk in rawh5['r5'] and nk!='fh5':
        diff = np.sum(rawh5['r5'][nk]-rawh5['r4'][nk])
        if diff>1e-5:
            raise ValueError('Difference observed in normalisation components in the two HDF5 norm files!')
            print(nk, diff)
#-----------------------------



#==========================================================
Cnt, txLUT, axLUT = nipet.sigaux.init_sig(rawh5['r4']['fh5'])
#==========================================================


#=============================================
# > GET THE TRANSIAXIAL LOR SIGN
sino = np.zeros((Cnt['NAW'], ), dtype=np.float32)
im = np.zeros((Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)

tv = np.zeros(Cnt['NTV']*Cnt['NAW'], dtype=np.uint8)
tt = np.zeros(Cnt['NTT']*Cnt['NAW'], dtype=np.float32)

nipet.prjsig.tprj(sino, im, tv, tt, txLUT, Cnt)
tt.shape = (Cnt['NAW'], Cnt['NTT'])
tt_ssgn = tt[:,4]

#matshow(np.reshape(tt_ssgn, (Cnt['NSBINS'], Cnt['NSANGLES'])).T)
#=============================================


# > repeat and reshape the geometric 16 slices corresponding to the detector unit width
nrm3d = rawh5['r4']['nrm3d'].copy()
for i in range(len(rawh5['r4']['sclfcr'])):
    nrm3d[i,...] *= rawh5['r4']['sclfcr'][i]

# > geometric norm sinogram
geosn = np.tile(nrm3d, [Cnt['NSANGLES']//nrm3d.shape[0],1,1])
geosn = np.transpose(geosn, (1,0,2))

# > crystal efficiencies
ceff = rawh5['r4']['ceff']
ceff = ceff[:,:Cnt['NCRS']]



#==========================================================
# > GENERATE THE NORM SINOGRAM
#==========================================================

# > initialisation of normalisation sinogram
nrmsn = np.zeros((Cnt['NSN'],Cnt['NSANGLES'],Cnt['NSBINS']),dtype=np.float32)

# > sinogram of crystal efficiencies represented as a vector
effsn = np.zeros(Cnt['NAW'], dtype=np.float32)

for li in range(len(axLUT['li2sn'])):
    
    # > sinogram index running linearly
    sni = axLUT['li2sn'][li,0]
    # > ring indices
    r0 = axLUT['li2rno'][li,0]
    r1 = axLUT['li2rno'][li,1]

    print(f'+> R0={r0}, R1={r1}, sni={sni}')

    effsn[:] = 0
    for bidx,(c0, c1) in enumerate(txLUT['s2c']):
        if tt_ssgn[bidx]>0.1:
            effsn[bidx] = ceff[r0,c0]*ceff[r1,c1]
        else:
            effsn[bidx] = ceff[r1,c0]*ceff[r0,c1]

    nrmsn[sni,...] = geosn[sni,...] * np.reshape(effsn, (Cnt['NSBINS'],Cnt['NSANGLES'])).T

    # > if direct sinogram, then skip the next step
    if li<Cnt['NRNG']: continue

    # > sinogram and ring indices
    sni = axLUT['li2sn'][li,1]
    r0 = axLUT['li2rno'][li,1]
    r1 = axLUT['li2rno'][li,0]

    print(f'-> R0={r0}, R1={r1}, sni={sni}')

    effsn[:] = 0
    for bidx,(c0, c1) in enumerate(txLUT['s2c']):
        if tt_ssgn[bidx]>0.1:
            effsn[bidx] = ceff[r0,c0]*ceff[r1,c1]
        else:
            effsn[bidx] = ceff[r1,c0]*ceff[r0,c1]

    if not sni in range(1,Cnt['NSEG0'],2):
        nrmsn[sni,...] = geosn[sni,...] * np.reshape(effsn, (Cnt['NSBINS'],Cnt['NSANGLES'])).T
    else:
        nrmsn[sni,...] = 0.25*(nrmsn[sni,...] + (geosn[sni,...] * np.reshape(effsn, (Cnt['NSBINS'],Cnt['NSANGLES'])).T))
#==========================================================

                
#nrm_ = nrmsn.transpose(1,0,2)

#si=23; matshow(100*(nrm_[:,si,:]-nrm[:,si,:])/nrm[:,si,:]); colorbar()