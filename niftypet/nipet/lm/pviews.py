#!/usr/bin/python
__author__ = 'pawel'

import numpy as np
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

def mvavg(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def video_frm(hst, outpth):

    plt.close('all')

    #=============== CONSTANTS ==================
    VTIME = 4
    #============================================

    i = np.argmax(hst['phc'])
    ymin = np.floor( min(hst['cmass'][i:i+300]) )
    ymax = np.ceil( max(hst['cmass'][i+100:]) )

    mfrm = hst['pvs_sgtl'].shape[0];

    #--for movie
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='GPU Sino Views', artist='Pawel', comment=':)')
    writer = FFMpegWriter(fps=25, bitrate=30000, metadata=metadata)
    #--

    fig3 = plt.figure()

    ax1 = plt.subplot(311)
    plt.title('Coronal View')
    plt.setp( ax1.get_xticklabels(), visible=False)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off')
    l1 = plt.imshow(hst['pvs_crnl'][100,:,:]/np.mean(hst['pvs_crnl'][100,:,:]), cmap='jet',interpolation='nearest')

    ax2 = plt.subplot(312)
    plt.title('Sagittal View')
    plt.setp( ax2.get_xticklabels(), visible=False)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off')
    l = plt.imshow(hst['pvs_sgtl'][100,:,:]/np.mean(hst['pvs_sgtl'][100,:,:]), cmap='jet',interpolation='nearest')

    ax3 = plt.subplot(313)
    plt.title('Axial Centre of Mass')
    t = np.arange(0., hst['dur'], 1.)
    #plt.plot(t, rprmt, 'k', t, rdlyd, 'r')
    plt.plot(t, mvavg(hst['cmass'][:],5),'k')
    plt.ylim([ymin, ymax])
    plt.xlabel('Time [s]')
    l2, = plt.plot(np.array([1000, 1000]), np.array([0, ymax]), 'b')

    #how many gpu frames per movie (controls the time resolution)
    mf = 6
    mmfrm = mfrm/mf

    fnm = os.path.join(outpth, 'pViews_' +str(mf)+'.mp4')

    with writer.saving( fig3, fnm, 200 ):
        for i in range(mmfrm):
            print('i> short frame to movie:', i)
            tmp = np.sum( hst['pvs_sgtl'][mf*i:mf*(i+1),:,:], axis=0)
            tmp2= np.sum( hst['pvs_crnl'][mf*i:mf*(i+1),:,:], axis=0)
            tmp = tmp/np.mean(tmp)
            tmp2 = tmp2/np.mean(tmp2)
            l.set_data(tmp)
            l1.set_data(tmp2)
            # l2.set_data(VTIME*mf*i*np.ones(2), np.array([0, np.max(hst['phc'])]))
            l2.set_data(VTIME*mf*i*np.ones(2), np.array([0, ymax]))
            writer.grab_frame()


    plt.show()
    return fnm

#===================================================================================
# Dynamic Frames to Projection Views
#-----------------------------------------------------------------------------------
def video_dyn(hst, frms, outpth, axLUT, Cnt):

    plt.close('all')

    #=============== CONSTANTS ==================
    VTIME = 4
    NRINGS = Cnt['NRNG']
    NSN11 = Cnt['NSN11']
    NDSN = Cnt['NSEG0']
    A = Cnt['NSANGLES']
    W = Cnt['NSBINS']

    voxz = Cnt['SO_VXZ']
    nsinos = NSN11
    #============================================

    # for scaling of the mass centre
    i = np.argmax(hst['phc'])
    ymin = np.floor( min(hst['cmass'][i:i+300]) )
    ymax = np.ceil( max(hst['cmass'][i+100:]) )

    # number of dynamic frames
    nfrm = hst['psino'].shape[0]
    # and cumulative sum of frame duration
    frmcum = np.cumsum(frms)

    # dynamic sino
    ddsino = np.zeros((nfrm, NDSN, A, W), dtype=np.int32)
    gsum = np.zeros(nfrm, dtype=np.int32)
    gpu_totsum = 0

    for frm in range(nfrm):
        for i in range(nsinos):
            ddsino[frm, axLUT['sn11_ssrb'][i], :, :] += hst['psino'][frm,i,:,:]
        gsum[frm] = np.sum(hst['psino'][frm,:,:,:])
        gpu_totsum += gsum[frm]
        print('GPU('+str(frm)+') =', gsum[frm])
        print('-----------')
    print('GPUtot =', gpu_totsum)

    #---additional constants
    saggital_angle = 127
    coronal_angle = 0
    i_mxfrm = gsum.argmax()
    frmrep = 5
    mfrm = frmrep*nfrm
    #---

    #--for movie
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Axial View', artist='Pawel', comment='--')
    writer = FFMpegWriter(fps=10, bitrate=30000, metadata=metadata)
    #--

    fig1 = plt.figure()

    ax1 = plt.subplot(311)
    plt.title('Coronal View')
    plt.setp( ax1.get_xticklabels(), visible=False)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off')
    l1 = plt.imshow(np.array(ddsino[i_mxfrm, : , coronal_angle, :], dtype=np.float64), cmap='jet',interpolation='nearest')
    #plt.clim([0, 70])

    ax2 = plt.subplot(312)
    plt.title('Sagittal View')
    plt.setp( ax2.get_xticklabels(), visible=False)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off')
    l2 = plt.imshow(np.array(ddsino[i_mxfrm, : , saggital_angle, :], dtype=np.float64), cmap='jet',interpolation='nearest')
    #plt.clim([0, 70])

    ax3 = plt.subplot(313)
    plt.title('Axial Centre of Mass')
    plt.plot(range(hst['dur']), voxz*mvavg(hst['cmass'][:],5),'k')
    plt.ylim([voxz*ymin, voxz*ymax])
    plt.xlabel('Time [s]')
    l3, = plt.plot(np.array([1000, 1000]), np.array([0, ymax]), 'b')

    fnm = os.path.join(outpth, 'pViews_dyn.mp4')
    with writer.saving(fig1, fnm, 100):
        for frm in range(mfrm):
            print ('i> dynamic frame:', frm%nfrm)
            tmp = np.array(ddsino[frm%nfrm, : , coronal_angle, :], dtype=np.float64)
            l1.set_data(tmp)
            tmp = np.array(ddsino[frm%nfrm, : , saggital_angle, :], dtype=np.float64)
            l2.set_data(tmp)
            l3.set_data(frmcum[frm%nfrm]*np.ones(2), np.array([0, ymax]))
            writer.grab_frame()

    return fnm

#===================================================================================