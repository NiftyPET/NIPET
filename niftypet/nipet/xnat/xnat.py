import numpy as np
import math
import os
import sys
import scipy.ndimage as ndi
import csv
import zipfile
import pydicom as dcm

import re
import nibabel as nib
from subprocess import call
import glob

#--------------
import pycurl
import json
from StringIO import StringIO
from urllib import urlencode
from datetime import datetime
#--------------

# DICOM extensions
dcm_ext = ('dcm', 'DCM', 'ima', 'IMA')


# # version of matplotlib to int
# import matplotlib.pyplot as plt
# import matplotlib
# mplv =  matplotlib.__version__.split('.')
# mplv = int(mplv[0])*10 + int(mplv[1])

# -----------------------------
def create_dir(pth):
    if not os.path.exists(pth):    
        os.makedirs(pth)
# -----------------------------

# -----------------------------
def time_stamp():
    now    = datetime.now()
    nowstr = str(now.year)+'-'+str(now.month)+'-'+str(now.day)+' '+str(now.hour)+':'+str(now.minute)
    return nowstr
# -----------------------------


#----------------------------------------------------------------------------------------------------------
def get_xnatList(xnaturi, cookie='', usrpwd=''):
    buffer = StringIO()
    c = pycurl.Curl()
    if cookie:
        c.setopt(pycurl.COOKIE, cookie)
    elif usrpwd:
        c.setopt(c.USERPWD, usrpwd)
    else:
        raise NameError('Session ID or username:password are not given')
    c.setopt(pycurl.SSL_VERIFYPEER, 0)
    c.setopt(pycurl.SSL_VERIFYHOST, 0)
    c.setopt(c.VERBOSE, 0)
    c.setopt(c.URL, xnaturi )
    c.setopt(c.WRITEDATA, buffer)
    c.perform()
    c.close()
    # convert to json dictionary in python
    outjson = json.loads( buffer.getvalue() )
    return outjson['ResultSet']['Result']

def get_xnat(xnaturi, frmt='json', cookie='', usrpwd=''):
    buffer = StringIO()
    c = pycurl.Curl()
    if cookie:
        c.setopt(pycurl.COOKIE, cookie)
    if usrpwd:
        c.setopt(c.USERPWD, usrpwd)
    else:
        raise NameError('Session ID or username:password are not given')
    c.setopt(pycurl.SSL_VERIFYPEER, 0)
    c.setopt(pycurl.SSL_VERIFYHOST, 0)
    c.setopt(c.VERBOSE, 0)
    c.setopt(c.URL, xnaturi )
    c.setopt(c.WRITEDATA, buffer)
    c.perform()
    c.close()
    # convert to json dictionary in python
    if frmt=='':
        output = buffer.getvalue()
    elif frmt=='json':
        output = json.loads( buffer.getvalue() )
    return output

def get_xnatFile(xnaturi, fname, cookie='', usrpwd=''):
    try:
        fn = open(fname, 'wb')
        c = pycurl.Curl()
        if cookie:
            c.setopt(pycurl.COOKIE, cookie)
        else:
            c.setopt(c.USERPWD, usrpwd)
        c.setopt(pycurl.SSL_VERIFYPEER, 0)
        c.setopt(pycurl.SSL_VERIFYHOST, 0)
        c.setopt(c.VERBOSE, 0)
        c.setopt(c.URL, xnaturi )
        c.setopt(c.WRITEDATA, fn)
        c.setopt(pycurl.FOLLOWLOCATION, 0)
        c.setopt(pycurl.NOPROGRESS, 0)
        c.perform()
        c.close()
        fn.close()
    except pycurl.error as pe:
        print ' '
        print '=============================================================='
        print 'e> pycurl error:', pe
        print '=============================================================='
        print ' '
        print 'w> no data:', sbjid, sbjlb
        return -1
    else:
        print '--------'
        print 'i> pycurl download done.'
        print '--------'
    return 0
#----------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------
def put_xnat(xnaturi, cookie='', usrpwd=''):
    """e.g., create a container"""
    c = pycurl.Curl()
    if cookie:
        c.setopt(pycurl.COOKIE, cookie)
    elif usrpwd:
        c.setopt(c.USERPWD, usrpwd)
    else:
        raise NameError('Session ID or username:password are not given')
    c.setopt(pycurl.SSL_VERIFYPEER, 0)
    c.setopt(pycurl.SSL_VERIFYHOST, 0)
    c.setopt(c.VERBOSE, 0)
    c.setopt(c.URL, xnaturi )
    c.setopt(c.CUSTOMREQUEST, 'PUT')
    c.perform()
    c.close()

def del_xnat(xnaturi, cookie='', usrpwd=''):
    """e.g., create a container"""
    c = pycurl.Curl()
    if cookie:
        c.setopt(pycurl.COOKIE, cookie)
    elif usrpwd:
        c.setopt(c.USERPWD, usrpwd)
    else:
        raise NameError('Session ID or username:password are not given')
    c.setopt(pycurl.SSL_VERIFYPEER, 0)
    c.setopt(pycurl.SSL_VERIFYHOST, 0)
    c.setopt(c.VERBOSE, 0)
    c.setopt(c.URL, xnaturi )
    c.setopt(c.CUSTOMREQUEST, 'DELETE')
    c.perform()
    c.close()

def post_xnat(xnaturi, post_data, verbose=0, PUT=False,  cookie='', usrpwd=''):
    buffer = StringIO()
    c = pycurl.Curl()
    if cookie:
        c.setopt(pycurl.COOKIE, cookie)
    elif usrpwd:
        c.setopt(c.USERPWD, usrpwd)
    else:
        raise NameError('Session ID or username:password are not given')
    c.setopt(c.USERPWD, usrpwd)
    c.setopt(pycurl.SSL_VERIFYPEER, 0)
    c.setopt(pycurl.SSL_VERIFYHOST, 0)
    c.setopt(c.VERBOSE, verbose)
    c.setopt(c.URL, xnaturi )
    if PUT: c.setopt(c.CUSTOMREQUEST, 'PUT')
    c.setopt(c.POSTFIELDS, post_data)
    c.setopt(c.WRITEFUNCTION, buffer.write)
    c.perform()
    c.close()
    return buffer.getvalue()

def put_xnatFile(xnaturi, filepath, cookie='', usrpwd=''):
    """upload file to xnat server"""
    c = pycurl.Curl()
    if cookie:
        c.setopt(pycurl.COOKIE, cookie)
    elif usrpwd:
        c.setopt(c.USERPWD, usrpwd)
    else:
        raise NameError('Session ID or username:password are not given')
    c.setopt(pycurl.SSL_VERIFYPEER, 0)
    c.setopt(pycurl.SSL_VERIFYHOST, 0)
    c.setopt(pycurl.NOPROGRESS, 0)
    c.setopt(c.VERBOSE, 0)
    c.setopt(c.URL, xnaturi )
    c.setopt(c.HTTPPOST, [('fileupload', (c.FORM_FILE, filepath,)),])
    c.perform()
    c.close()
#----------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------
def put_xnatPetMrRes(usrpwd, xnatsbj, sbjix, lbl, frmt, fpth):
    # get the experiment id
    expt = get_xnatList(usrpwd, xnatsbj+'/' +sbjix+ '/experiments?xsiType=xnat:petmrSessionData&format=json&columns=ID')
    # prepare the uri
    xnaturi = xnatsbj+'/' +sbjix+ '/experiments/' + expt[0]['ID'] + '/resources/' +lbl+ '?xsi:type=xnat:resourceCatalog&format='+frmt
    xnat_put(usrpwd, xnaturi)
    # upload
    xnaturi = xnatsbj+'/' +sbjix+ '/experiments/' + expt[0]['ID'] + '/resources/' +lbl+ '/files'
    xnat_upload(usrpwd, xnaturi, fpth)
#----------------------------------------------------------------------------------------------------------




#====================================================================================================
def download(xc, sbjid, sbjno,
    tpoint=0,
    expmts=[],
    expid=[],
    download=False,
    prcl=False,
    t1w=False,
    t2w=False,
    umap=False,
    ute=False,
    pct=False,
    nrm=False,
    lm=False,
    cookie=''):

    '''
    expmts: list of dictionaries for XNAT experiments
    expid: [idnex for expmts or expt below when expmts is not given, time point label ordered according to the date]
    tpoint: time point according to date of scan, starting from 0 for baseline
    etc...
    '''

    # subject index in XNAT server
    sbjix = xc['prjf']+ '_' + sbjno

    # get the experiment ID
    if not expmts:
        try:
            if cookie:
                expmts = get_xnatList(
                    xc['sbjs']+'/' +sbjix+ '/experiments?xsiType=xnat:petmrSessionData&format=json',
                    cookie=cookie)
            else:
                expmts = get_xnatList(
                    xc['sbjs']+'/' +sbjix+ '/experiments?xsiType=xnat:petmrSessionData&format=json',
                    usrpwd=xc['usrpwd'])

        except ValueError as ve:
            print '-------------------'
            print 'e> Value Error:', ve
            print '   for subject:', sbjix
            print '-------------------'
            raise

    #import pdb; pdb.set_trace()

    if expid:
        tpstr = 'TP'+str(expid[1])
    else:
        # dates for available time point scans
        dates = [datetime.strptime(d['date'], "%Y-%m-%d") for d in expmts]
        # date index for sorting
        didx = [k[0] for k in sorted(enumerate(dates), key=lambda x:x[1])]
        tpstr = 'TP'+str(tpoint)
        try:
            expid = [didx.index(np.int(tpoint)), tpoint]
        except ValueError:
            print 'e> time point {} does not exists!'.format(str(tpoint))
            return None

    # get the right experiment
    expt = expmts[expid[0]]
    if not expt:
        print 'w> no data for:', sbjid, sbjno
        return None
        

    # core subject output path
    spth = os.path.join( xc['opth'], os.path.join(sbjno+'_'+sbjid, tpstr ) )

    # norm path
    npth = os.path.join(spth, 'norm')
    # LM path
    lmpth = os.path.join(spth, 'LM')
    # T1/2 folder
    t1pth = os.path.join(spth, 'T1w')
    t2pth = os.path.join(spth, 'T2w')
    # pseudo-CT folder
    pctpth = os.path.join(spth, 'pCT')
    # DCM mu-map
    umpth = os.path.join(spth, 'umap')
    # UTE two sequences
    utepth = os.path.join(spth, 'UTE')
    # regional parcellations
    parpth = os.path.join(spth, 'prcl')

    # output dictionary
    out = {}

    # get all scans
    scans = get_xnatList(
        xc['sbjs']+'/' +sbjix+ '/experiments/' + expt['ID'] + '/scans', cookie=cookie
    )
    scntypes = [(s['type'],s['quality'],s['ID']) for s in scans]
    
    # import pdb; pdb.set_trace()

    # get the list of T1w files
    if t1w:
        out['T1nii'] = []
        if isinstance(t1w, basestring):
            pick_types = [s for s in scntypes if t1w == s[0]]
        if isinstance(t1w, list):
            pick_types = []
            for tt in t1w:
                pick_types += [s for s in scntypes if tt == s[0]]
        else:
            pick_types = [s for s in scntypes if 't1' in s[0].lower()]

        #-go through all scan types
        for sel_type in pick_types:
            TYPE = sel_type[0]
            QLTY = sel_type[1]
            ID = sel_type[2]
            t1files = get_xnatList(
                xc['sbjs']+'/' +sbjix+ '/experiments/' + expt['ID'] + '/scans/'+ID+'/resources/NIFTI/files',
                cookie=cookie
            )
            if download:
                opth = t1pth
                create_dir(opth)
                for i in range(len(t1files)):
                    # add 't1_' for the file to be recognised as T1w
                    fname = 't1_scan-'+ID+'_'+QLTY+'_'+TYPE+'_'+sbjno+'_'+sbjid+'.'+t1files[i]['Name'].split('.',1)[-1]
                    status = get_xnatFile(
                        xc['url']+t1files[i]['URI'],
                        os.path.join(opth, fname),
                        cookie=cookie)
                    if status<0:
                        print 'e> no T1w image data:', sbjid, sbjno
                    else:
                        out['T1nii'].append(os.path.join(opth, fname))
                if len(t1files)<1: 
                    print 'e> no T1w image data:', sbjid, sbjno
            else:
                out['T1nii'] += len(t1files)
    # --------------------------------------------------------------------------------
    if t2w:
        out['T2nii'] = []
        if isinstance(t2w, basestring):
            pick_types = [s for s in scntypes if t2w == s[0]]
        if isinstance(t2w, list):
            pick_types = []
            for tt in t2w:
                pick_types += [s for s in scntypes if tt == s[0]]
        else:
            pick_types = [s for s in scntypes if 't1' in s[0].lower()]

        #-go through all scan types
        for sel_type in pick_types:
            TYPE = sel_type[0]
            QLTY = sel_type[1]
            ID = sel_type[2]
            files = get_xnatList(
                xc['sbjs']+'/' +sbjix+ '/experiments/' + expt['ID'] + '/scans/'+ID+'/resources/NIFTI/files',
                cookie=cookie
            )
            if download:
                opth = t2pth
                create_dir(opth)
                for i in range(len(files)):
                    # add 't2_' for the file to be recognised as T1w
                    fname = 't2_scan-'+ID+'_'+QLTY+'_'+TYPE+'_'+sbjno+'_'+sbjid+'.'+files[i]['Name'].split('.',1)[-1]
                    status = get_xnatFile(
                        xc['url']+files[i]['URI'],
                        os.path.join(opth, fname),
                        cookie=cookie)
                    if status<0:
                        print 'e> no T1w image data:', sbjid, sbjno
                    else:
                        out['T1nii'].append(os.path.join(opth, fname))
                if len(files)<1: 
                    print 'e> no T1w image data:', sbjid, sbjno
            else:
                out['T2nii'] += len(files)
    # --------------------------------------------------------------------------------

    if ute:
        out['UTE1'], out['UTE2'] = [], []
        if isinstance(ute, basestring):
            pick_types = [s for s in scntypes if ute == s[0]]
        elif isinstance(ute, list):
            pick_types = []
            for tt in ute:
                pick_types += [s for s in scntypes if tt == s[0]]
        else:
            pick_types = [s for s in scntypes if 'ute' in s[0].lower()]

        print 'i> picked types:', pick_types

        for sel_type in pick_types:
            TYPE = sel_type[0]
            QLTY = sel_type[1]
            ID = sel_type[2]
            files = get_xnatList(
                xc['sbjs']+'/' +sbjix+ '/experiments/' + expt['ID'] + '/scans/'+ID+'/resources/DICOM/files',
                cookie=cookie
            )
            if download:
                create_dir(utepth)
                for i in range(len(files)):
                    status = get_xnatFile(
                        xc['url']+files[i]['URI'],
                        os.path.join(utepth, files[i]['Name']),
                        cookie=cookie
                    )
                    if status<0:
                        print 'e> no UTE data:', sbjid, sbjno
                        raise IOError('Could not cownload the UTE data')
                        
                if len(files)<192 and [n['Name'] for n in files if '.zip' in n['Name']]:
                    fnm = [n['Name'] for n in files if '.zip' in n['Name']][0]
                    fzip = zipfile.ZipFile(os.path.join(utepth,fnm), 'r')
                    fzip.extractall(utepth)
                    fzip.close()
                # get the number of DICOM files
                out['#UTE'] = len([f for f in os.listdir(utepth) if f.endswith(dcm_ext)])
                # read one of the DICOM files to get the echo time
                fdcm = glob.glob(os.path.join(utepth, '*.dcm'))[0]
                dhdr = dcm.dcmread(fdcm)
                # get the Echo time of the UTE sequences
                if [0x018, 0x081] in dhdr:
                    ETstr = str(dhdr[0x018, 0x081].value)
                    # target path depending on the echo time
                    trgtpth = os.path.join(os.path.dirname(utepth), 'UTE_scan-'+ID+'_Q-'+QLTY+'_ET-'+ETstr.replace('.','-'))
                    os.rename(utepth,  trgtpth)
                else:
                    raise KeyError('Could  not find Echo Time in the DICOM header')
                if np.float32(ETstr)<0.1:
                    out['UTE1'] = trgtpth
                else:
                    out['UTE2'] = trgtpth

                        

    # DCM mu-map (UTE)
    if umap:
        um_TYPE = '1946_MRAC_UTE_UMAP'
        # get the list of norm files
        umfiles = get_xnatList(
            xc['sbjs']+'/' +sbjix+ '/experiments/' + expt['ID'] + '/scans/'+um_TYPE+'/resources/DICOM/files',
            cookie=cookie
        )
        # download the norm files:
        out['UTE'] = []
        if download:
            create_dir(umpth)
            for i in range(len(umfiles)):
                status = get_xnatFile(
                    xc['url']+umfiles[i]['URI'],
                    os.path.join(umpth, umfiles[i]['Name']),
                    cookie=cookie
                )
                if status<0:
                    print 'e> no UTE/Dixon mu-map data:', sbjid, sbjno
                else:
                    out['UTE'].append(os.path.join(umpth, umfiles[i]['Name']))
            if len(umfiles)<192:
                if any(n['Name'] for n in umfiles if 'zip' in n['Name']):
                    fzip = zipfile.ZipFile(out['UTE'][0], 'r')
                    fzip.extractall(os.path.dirname(out['UTE'][0]))
                    fzip.close()
                    out['#UTE'] = len(os.listdir(os.path.dirname(out['UTE'][0])))
                else:
                    print 'e> UTE image data too small (<10):', sbjid, sbjno
                    out['UTE'] = 'missing'
            elif len(umfiles)==192:
                out['#UTE'] = 192
        else:
            out['UTE'] = str(len(umfiles))

    #--------------------------------------------------------------------------------------
    # brain parcellations
    if prcl:
        PAR_TYPE = 'GIF_v2_TEST'
        parfiles = get_xnatList(
            xc['sbjs']+'/' +sbjix+ '/experiments/' + expt['ID'] + '/reconstructions/'+PAR_TYPE+'/files',
            cookie=cookie)
        if parfiles: out['prcl'] = []
        if download:
            create_dir(parpth)
            for i in range(len(parfiles)):
                status = get_xnatFile(
                    xc['url']+parfiles[i]['URI'], os.path.join(parpth, parfiles[i]['Name']),
                    cookie=cookie)
                if status<0:
                    print 'e> no parcellation image data:', sbjid, sbjno
                else:
                    out['prcl'].append( os.path.join(parpth,parfiles[i]['Name']) )
            if len(parfiles)<5: 
                print 'e> it seems there is not enough parcellation data for:', sbjid, sbjno
        else:
            out['prcl'] = str(len(parfiles))


    # pseudo-CT
    if pct:
        PCT_TYPE = 'PCT_HU'
        pctfiles = get_xnatList(
            xc['sbjs']+'/' +sbjix+ '/experiments/' + expt['ID'] + '/resources/'+PCT_TYPE+'/files',
            cookie=cookie)
        if download:
            create_dir(pctpth)
            for i in range(len(pctfiles)):
                status = get_xnatFile(
                    xc['url']+pctfiles[i]['URI'], os.path.join(pctpth, pctfiles[i]['Name']),
                    cookie=cookie)
                if status<0:
                    print 'e> no pCT image data:', sbjid, sbjno
                else:
                    out['pCT'] = os.path.join(pctpth, pctfiles[i]['Name'])
            if len(pctfiles)<1: 
                print 'e> missing pCT file for', sbjid, sbjno
        else:
            out['pCT'] = str(len(pctfiles))


    # get the list of norm files
    if nrm:
        nrmfiles = get_xnatList(
            xc['sbjs']+'/' +sbjix+ '/experiments/' + expt['ID'] + '/resources/Norm/files',
            cookie=cookie
        )
        # download the norm files:
        if download:
            create_dir(npth)
            for i in range(len(nrmfiles)):
                status = get_xnatFile(
                    xc['url']+nrmfiles[i]['URI'], os.path.join(npth, nrmfiles[i]['Name']),
                    cookie=cookie
                )
                if status<0:
                    if ip>=0: del plist[ip]
                    print 'e> no NORM data:', sbjid, sbjno
                else:
                    if '.dcm' in nrmfiles[i]['Name'].lower():
                        out['nrm_dcm'] = os.path.join(npth, nrmfiles[i]['Name'])
                    elif '.bf' in nrmfiles[i]['Name'].lower():
                        out['nrm_bf'] = os.path.join(npth, nrmfiles[i]['Name'])
                    elif '.ima' in nrmfiles[i]['Name'].lower():
                        out['nrm_ima'] = os.path.join(npth, nrmfiles[i]['Name'])
            if len(nrmfiles)<1:
                print 'e> norm data missing.'
        else:
            out['norm'] = str(len(nrmfiles))

    # get the list of LM files
    if lm:
        lmfiles = get_xnatList(
            xc['sbjs']+'/' +sbjix+ '/experiments/' + expt['ID'] + '/resources/LM/files',
            cookie=cookie
        )
        # download the LM files:
        if download:
            create_dir(lmpth)
            for i in range(len(lmfiles)):
                # check if the file is already downloaded:
                if  os.path.isfile ( os.path.join(lmpth, lmfiles[i]['Name']) ) and \
                    str(os.path.getsize(os.path.join(lmpth, lmfiles[i]['Name'])))==lmfiles[i]['Size']:
                    print 'i> file of the same size,',lmfiles[i]['Name'], 'already exists: skipping download.'
                    if '.dcm' in lmfiles[i]['Name'].lower():
                        out['lm_dcm'] = os.path.join(lmpth, lmfiles[i]['Name'])
                    elif '.bf' in lmfiles[i]['Name'].lower():
                        out['lm_bf'] = os.path.join(lmpth, lmfiles[i]['Name'])
                    elif '.ima' in lmfiles[i]['Name'].lower():
                        out['lm_ima'] = os.path.join(lmpth, lmfiles[i]['Name'])
                else:
                    status = get_xnatFile(
                        xc['url']+lmfiles[i]['URI'], os.path.join(lmpth, lmfiles[i]['Name']),
                        cookie=cookie
                    )
                    if status<0:
                        print 'w> no LM data:', sbjid, sbjno
                    else:
                        if '.dcm' in lmfiles[i]['Name'].lower():
                            out['lm_dcm'] = os.path.join(lmpth, lmfiles[i]['Name'])
                        elif '.bf' in lmfiles[i]['Name'].lower():
                            out['lm_bf'] = os.path.join(lmpth, lmfiles[i]['Name'])
                        elif '.ima' in lmfiles[i]['Name'].lower():
                            out['lm_ima'] = os.path.join(lmpth, lmfiles[i]['Name'])
            if len(lmfiles)<1:
                print 'e> LM data missing.'
        else:
            out['LM'] = str(len(lmfiles))

    out['folderin'] = spth

    return out













# #<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# def dobtp(xc, sbjid, sbjno, t0, t1, Nr, gpth, labels, dkeys, rois, txLUT, axLUT, Cnt):
#     # xc: dictionary of all properties for accessing XNAT server
#     # sbjid: subject ID
#     # sbjno: subject number
#     # t0: frame start time
#     # t1: frame stop time
#     # Nr: number of bootstrap realisations
#     # gpth: path to GIF parcellations
#     # labels: ROI labels based on the provided parcellation
#     # dkeys: keys (labels) used in the dictionary of the ROIs (easier/shorter for programming than the above labels)
#     # rois: the actual ROIs defined in numbers
#     # txLUT, axLUT, Cnt: transaxial, axial LUTs and scanner (Siemens mMR) constants

#     # subject index in XNAT server
#     sbjix = xc['prjf']+ '_' + sbjno

#     # core subject output path
#     spth = os.path.join(xc['opth'], sbjno+'_'+sbjid)
#     # processing output path
#     outpth = os.path.join(spth, 'output')
#     # LM path
#     lmpth = os.path.join(spth, 'LM')
    
#     # output file names and their xnat content tags
#     btpresults = [['SEimg_btp_'+str(Nr)+'.nii.gz',
#                    'Rimg_btp_' +str(Nr)+'.nii.gz',
#                    'bootstrap_boxplot_btp_'+str(Nr)+'.png'],
#                    ['SEBIMG','RBIMG','BTPBXP']]


#     #------------------------------------------------------
#     # check if the parcellation is available
#     g_lb = glob.glob( os.path.join(gpth, sbjid+'*labels.nii*') )
#     g_bc = glob.glob( os.path.join(gpth, sbjid+'*bc.nii*') ) 
#     if not (g_bc and g_lb):
#         print 'e> no GIF parcellation files available'
#         return {'pliststr':'m:par'}

#     # split the label file path
#     lbpth = os.path.split(g_lb[0])

#     # sum of all voxel values in any given ROI
#     roisum = {}
#     for k in dkeys: roisum[k] = [] 

#     # sum of the mask values <0,1> used for getting the mean value of any ROI
#     roimsk = {}
#     for k in dkeys: roimsk[k] = []
#     #------------------------------------------------------


#     #------------------------------------------------------
#     #GET THE RAW & INPUT DATA FOR RECONSTRUCTION
#     folderin = spth
#     datain = nipet.mmraux.explore_input(folderin, Cnt)
#     #------------------------------------------------------

#     #print the time difference between calibration and acquisition
#     nipet.mmraux.time_diff_norm_acq(datain)


#     # =============================================================
#     # R E C O N    W I T H    R O I    S A M P L I N G
#     # -------------------------------------------------------------
#     if Cnt['SPN']==11:
#         snshape = (Cnt['NSN11'], Cnt['NSANGLES'], Cnt['NSBINS'])
#     elif Cnt['SPN']==1:
#         snshape = (Cnt['NSN1'], Cnt['NSANGLES'], Cnt['NSBINS'])
#     print 'i> using this sino shape:', snshape

#     # for calculating variance online of scatter and random sinos
#     M1_ssn = np.zeros(snshape, dtype=np.float32)
#     M2_ssn = np.zeros(snshape, dtype=np.float32)

#     M1_sss = np.zeros(snshape, dtype=np.float32)
#     M2_sss = np.zeros(snshape, dtype=np.float32)

#     M1_rsn = np.zeros(snshape, dtype=np.float32)
#     M2_rsn = np.zeros(snshape, dtype=np.float32)

#     M1_img = np.zeros((Cnt['SO_IMZ'],Cnt['SO_IMY'],Cnt['SO_IMX']), dtype=np.float32)
#     M2_img = np.zeros((Cnt['SO_IMZ'],Cnt['SO_IMY'],Cnt['SO_IMX']), dtype=np.float32)

#     # sys.exit()


#     # do single non-bootstrap or multiple bootstrap (Nr) realisations
#     for itr in range((Nr-1)*(Cnt['BTP']>0)+1):

#         print ''
#         print '++++++++++++++++++++++++++++'
#         print 'i> performing realisation:', itr
#         print '++++++++++++++++++++++++++++'
#         print ''

#         # --------------------------------------------------------------
#         # histogram the LM data and save the results
#         hst = nipet.lm.mmrhist.hist(datain, txLUT, axLUT, Cnt, hst_store=False, t0=t0, t1=t1)

#         #get the mu-maps (hardware and object);  use the comment to save each affine transformation text file
#         cmmnt = '_btp'+str(Cnt['BTP'])+'_'+str(itr)
#         mumaps = nipet.img.mmrimg.get_mumaps(datain, Cnt, hst=hst, t0=t0, t1=t1, fcomment=cmmnt)

#         #update datain
#         datain = nipet.mmraux.explore_input(folderin, Cnt)

#         # do the proper recon with attenuation and scatter
#         recon = nipet.prj.mmrprj.osemone(datain, mumaps, hst, txLUT, axLUT, Cnt, recmod=3, itr=4, fwhm=0., store_img=True, ret_sct=True)
#         # name of saved file
#         fpetq = recon[1]
#         # reconstructed image
#         imcrr = recon[0]
#         # if bootstrap, calculate variance online
#         if Cnt['BTP']>0:
#             # calculate variance online:
#             nipet.mmr_auxe.varon(M1_ssn, M2_ssn, recon[3], itr, Cnt)
#             nipet.mmr_auxe.varon(M1_sss, M2_sss, recon[4], itr, Cnt)
#             nipet.mmr_auxe.varon(M1_rsn, M2_rsn, recon[6], itr, Cnt)
#             nipet.mmr_auxe.varon(M1_img, M2_img, imcrr, itr, Cnt)
#         # --------------------------------------------------------------


#         # --------------------------------------------------------------
#         # GIF: get the T1w and labels in register with PET
#         print '';  print ''
#         print 'i> extracting ROI values:'
#         # T1w:
#         t1dir = os.path.dirname(datain['T1nii'])
#         f_ = glob.glob( os.path.join(t1dir, '*_affine'+cmmnt+'.txt') )
#         if len(f_)>0:
#             frigid = f_[0]
#             print 'i> using rigid transform from this file:', frigid
#         else:
#             print 'e> no affine transformation text file found!'
#             return {'pliststr':'m:--'}

#         # PET (without scatter and attenuation corrections):
#         petdir = os.path.dirname(datain['em_nocrr'])
#         fpet = glob.glob( os.path.join(petdir, '*__ACbed.nii.gz') )
#         if len(fpet)==0:
#             print 'w> NAC PET image not found: using quantitative PET image for registration of MR to PET.'
#             fpet = fpetq
#         elif len(fpet)>1:
#             print 'w> multiple NAC/NSCT PET images found: using quantitative PET image for registration of MR to PET.'
#             fpet = fpetq
#         # resample the GIF T1/labels
#         #call the resampling routine to get the GIF T1/labels in PET space
#         if len(g_lb)>0 and len(fpet)>0:
#             fgt1 = os.path.join(t1dir, os.path.basename(g_lb[0]).split('.')[0]+ '_toPET' +'.nii.gz')
#             if os.path.isfile( Cnt['RESPATH'] ):
#                 call(
#                     [Cnt['RESPATH'],
#                     '-ref', fpet[0],
#                     '-flo', g_bc[0],
#                     '-trans', frigid,
#                     '-res', fgt1])
#             else:
#                 print 'e> path to resampling executable is incorrect!'
#                 sys.exit()

#         if len(g_lb)==0 or len(fpet)==0:
#             print 'e> no enough input data for resampling (e.g., floating labels or target PET image)!'
#             return {'pliststr':'m:--'}

#         # Get the labels before resampling to PET space
#         nilb = nib.load(g_lb[0])
#         A = nilb.get_sform()
#         imlb = nilb.get_data()
#         # get a copy of the image for masks (still in the original T1 orientation)
#         roi_img = np.copy(imlb)

#         for k in rois.keys():
#             roi_img[:] = 0
#             for i in rois[k]:
#                 roi_img[imlb==i] = 1.
#             # now save the image mask to nii file <froi1>
#             froi1 = os.path.join(t1dir, lbpth[1].split('.')[0][:8]+'_'+k+'.nii.gz')
#             nii_roi = nib.Nifti1Image(roi_img, A)
#             nii_roi.header['cal_max'] = 1.
#             nii_roi.header['cal_min'] = 0.
#             nib.save(nii_roi, froi1)
#             # file name for the resampled ROI to PET space
#             froi2 = os.path.join(t1dir, os.path.basename(g_lb[0]).split('.')[0]+ '_toPET_'+k+'.nii.gz')
#             if os.path.isfile( Cnt['RESPATH'] ):
#                 call(
#                     [Cnt['RESPATH'],
#                     '-ref', fpet[0],
#                     '-flo', froi1,
#                     '-trans', frigid,
#                     '-res', froi2])
#             else:
#                 print 'e> path to resampling executable is incorrect!'
#                 sys.exit()
#             # get the resampled ROI mask image
#             rmsk = nimpa.prc.getnii(froi2)
#             rmsk[rmsk>1.] = 1.
#             rmsk[rmsk<0.] = 0.

#             # erode the cerebral white matter region
#             if k=='EroWM':
#                 nilb = nib.load(froi2)
#                 B = nilb.get_sform()
#                 # tmp = nilb.get_data()
#                 tmp = ndi.filters.gaussian_filter(rmsk, nipet.mmraux.fwhm2sig(12./10), mode='mirror')
#                 rmsk = np.float32(tmp>0.85)
#                 froi3 = os.path.join(t1dir, os.path.basename(g_lb[0]).split('.')[0]+ '_toPET_'+k+'_eroded.nii.gz')
#                 nipet.img.mmrimg.savenii(rmsk, froi3, B, Cnt)

#             # ROI value and mask sums:
#             rvsum  = np.sum(imcrr*rmsk)
#             rmsum  = np.sum(rmsk)
#             roisum[k].append( rvsum )
#             roimsk[k].append( rmsum )
#             print ''
#             print '================================================================'
#             print 'i> ROI extracted:', k
#             print '   > value sum:', rvsum
#             print '   > mask sum :', rmsum
#             print '================================================================'
#         # --------------------------------------------------------------

#     # =============================================================

#     if Cnt['BTP']>0:
#         np.save(os.path.join(outpth,'saved_roisum_btp'+str(itr+1)+'.npy'), roisum)
#         np.save(os.path.join(outpth,'saved_roimsk_btp'+str(itr+1)+'.npy'), roimsk)
#         np.save(os.path.join(outpth,'saved_varon_btp'+str(itr+1)+'.npy'), (M1_rsn, M1_sss, M1_ssn, M2_rsn, M2_sss, M2_ssn, itr))
#         np.save(os.path.join(outpth,'saved_varon_img_btp'+str(itr+1)+'.npy'), (M1_img, M2_img, itr))
#     else:
#         np.save(os.path.join(outpth,'saved_roisum.npy'), roisum)
#         np.save(os.path.join(outpth,'saved_roimsk.npy'), roimsk)


#     # ANALYSIS
#     # get the ROIs for reference and neocortex regions
#     ROIs = np.zeros((len(dkeys),itr+1), dtype=np.float32)
#     for i in range(len(dkeys)):
#         ROIs[i,:] = np.array(roisum[dkeys[i]]) / np.array(roimsk[dkeys[i]])

#     # pick the reference regions for reporting
#     refi = [0,2,3]
#     # pick the neocortex regions for reporting (boxplot)
#     roii = range(7, len(dkeys))
#     roilables = [labels[k] for k in roii]
#     print 'i> chosen regions of interests:',[labels[i] for i in refi]
#     print 'i> chosen regions of interests:', roilables


#     # one plot per ROI
#     if Cnt['BTP']>0:
#         ip = 1
#         plt.figure(figsize=(16, 12))
#         for i in range(len(refi)):
#             print 'i> ref region:', labels[refi[i]]
#             roi = np.array(roisum[dkeys[refi[i]]]) / np.array(roimsk[dkeys[refi[i]]])
#             roi.shape = (1, itr+1)
#             roi = np.repeat(roi, len(roii), axis=0)
#             suvr = ROIs[roii,:]/roi
#             plt.subplot(1,3,ip)
#             if mplv>15:       plt.boxplot(suvr.T, showmeans=True, labels=roilables)
#             else:             plt.boxplot(suvr.T, labels=roilables)
#             plt.axhline(y=suvr[-1,:].mean(),linewidth=1, color='b')
#             plt.xticks(range(1,1+len(roii)), rotation='vertical')
#             plt.title(labels[refi[i]]+' as reference')
#             plt.subplots_adjust(bottom=0.20)
#             ip += 1
#         plt.savefig(os.path.join(outpth, btpresults[0][2]), format='png', dpi=300)

#         # ratio (aka. SUVR) and SE image
#         Rimg = M1_img/np.mean(ROIs[0,:])
#         SEimg = ( (M2_img/itr)**.5 ) / np.mean(ROIs[0,:])
#         nipet.img.mmrimg.savenii(SEimg, os.path.join(outpth, btpresults[0][0]), Cnt['AFFINE'], Cnt)
#         nipet.img.mmrimg.savenii(Rimg,  os.path.join(outpth, btpresults[0][1]), Cnt['AFFINE'], Cnt)

#         # write ROIs to csv file
#         np.savetxt( os.path.join(outpth,'bootstrap_ROIsum.csv'), ROIs.T, delimiter=',', header=','.join(fieldnames) )


#     # save the boxplot and the NIfTI images to XNAT
#     if xc['upload']:
#         xnatURL = xc['sbjs']+'/' +sbjix+ '/experiments/' + expt[0]['ID'] + '/resources/BTP'
#         nipet.xnat.qc_xnat.put_xnat(xc['usrpwd'], xnatURL+'?xsi:type=xnat:resourceCatalog&format=PNG')
#         nipet.xnat.qc_xnat.put_xnatFile(xc['usrpwd'], xnatURL+'/files?content='+btpresults[1][2], os.path.join(outpth, btpresults[0][2]) )

#         nipet.xnat.qc_xnat.put_xnat(xc['usrpwd'], xnatURL+'?xsi:type=xnat:resourceCatalog&format=NIFTI')
#         nipet.xnat.qc_xnat.put_xnatFile(xc['usrpwd'], xnatURL+'/files?content='+btpresults[1][1], os.path.join(outpth, btpresults[0][1]) )
#         nipet.xnat.qc_xnat.put_xnatFile(xc['usrpwd'], xnatURL+'/files?content='+btpresults[1][0], os.path.join(outpth, btpresults[0][0]) )

#     dic_out = { 'pliststr':'y',
#                 'roisum':roisum, 'roimsk':roimsk, 'M1_rsn':M1_rsn, 'M1_sss':M1_sss, 'M1_ssn':M1_ssn, 'M2_rsn':M2_rsn,
#                 'M2_sss':M2_sss, 'M2_ssn':M2_ssn, 'M1_img':M1_img, 'M2_img':M1_img, 'itr':itr, 'ROIs':ROIs, 'labels':labels }
    
#     #remove the downloaded LM file
#     if xc['rmvlm']:   os.remove(os.path.join(lmpth, lmfiles[i]['Name']))

#     return dic_out


