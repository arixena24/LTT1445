# -*- coding: utf-8 -*-
# """
# Created on Wed Sep  9 09:44:11 2020
# @author: Trevor
# """
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import optimize, signal
import pickle
import warnings
warnings.filterwarnings("ignore")

def wfc3_extract(hist_med, plnm='W79', numxaps=5, numyaps=10):

#---Working Directory----------------------------------------------   
    fpath = '/Users/mauralally/Desktop/HST_project/' 
    filenames = glob.glob(fpath + 'LTT1445_test/' + '*flt_test.fits')
    filenames = sorted(filenames)

#---LOOP OVER EACH FITS FILE---------------------------------------
    
    for i in range(0, len(filenames)):
        hdulist = fits.open(filenames[i])

#-------make the arrays we will need-------------------------------
        if i == 0:
            img_ax = np.asarray(hdulist[1].data)
            subarray=len(img_ax)
            subarray2 = len(img_ax[0])
            nsamp=1
            print (str(len(filenames))+'files and'+str(nsamp)+'samples each')
            nsamp=int(nsamp)
            temp=np.zeros(shape=(subarray, subarray2))
            images=np.zeros(shape=(subarray, subarray2, len(filenames)))
            time=np.zeros(len(filenames))
            raDeg=np.zeros(len(filenames))
            decDeg=np.zeros(len(filenames))
            wlc=np.zeros(shape=(len(filenames), numyaps, numxaps))
            scan_ang=np.zeros(len(filenames))
            ndrwhite=np.zeros(shape=(numyaps, numxaps))  
            flux=np.zeros(shape=(numyaps, numxaps))
            error=np.zeros(shape=(numyaps, numxaps))
            diff=np.zeros(shape=(numyaps, numxaps))
            wlc_error=np.zeros(shape=(len(filenames), numyaps, numxaps))
        else: 
            pass
                    
#-------get info from headers----------------------------------------
        scidata=hdulist[1].data
        err=hdulist[2].data
        
        expstart=hdulist[0].header['EXPSTART']
        expend=hdulist[0].header['EXPEND']
        exptime=hdulist[0].header['EXPTIME']
        scan_ang[i]=hdulist[0].header['SCAN_ANG']
        scanlen=hdulist[0].header[ 'SCAN_LEN']
        raDeg = hdulist[0].header['RA_TARG']
        decDeg = hdulist[0].header['DEC_TARG']
        time[i]=0.5*(expend+expstart)
        if i ==0:
            xybox=getbox(scidata) #THIS GETS A BOX THE SIZE OF THE SPEC
            x_range=xybox[1]+1-xybox[0]
            y_range=xybox[3]+1-xybox[2]
            x_cen=np.floor((xybox[1]+1-xybox[0])/2.)
            y_cen=np.floor((xybox[3]+1-xybox[2])/2.)

#-------flat field and background--------------------------------------
        scidata, images = background_and_flat(scidata, images, hist_med, i)
    
        for aprx in range(0, numxaps):
            xwidth=x_range+aprx
            
            for apry in range(0, numyaps):
# #                 ff=np.sum(scidata[xybox[2]-apry:xybox[3]+1+apry, xybox[0]-aprx:xybox[1]+1+aprx])
                er=np.sum(err[xybox[2]-apry:xybox[3]+1+apry, xybox[0]-aprx:xybox[1]+1+aprx]**2.)
#                 flux[apry, aprx] = wlc
                error[apry, aprx]=er
        wlc_error[i,:,:]=np.sum(error, axis=0)**0.5
        diff=np.zeros(shape=(numyaps, numxaps))
        wlc[i, :, :]=diff
        
    print('wlc', wlc)
    plt.figure()
    plt.errorbar(time,  wlc[:,0,0], yerr=wlc_error[:,0,0], fmt='o', color='k')
    plt.xlabel('Time ($MJD_{UTC}$)')
    plt.ylabel('Flux (e$^-$)')
    plt.title('WFC3 Raw Light curve')
    plt.show()
    
    outpath = fpath+'LTT1445_outpath/'
    fileObject = open(outpath+'wlc_extract_out', 'wb')
    pickle.dump([time, wlc, wlc_error, raDeg, decDeg, scidata, 
                 xybox], fileObject)
    fileObject.close()
    
    return time, wlc, wlc_error, raDeg, decDeg, scidata, xybox


def background_and_flat(scidata, images, hist_med, i):
#---define columns at the edge of the image to mask--------------------------------
#     cols1 = np.arange(0,50)
#     cols2 = np.arange(450, 512)
#     edges=np.append(cols1,cols2)
#     m = np.zeros_like(scidata)
#     m[:,edges] = 1
#     m[edges, :] = 1
#     scidata=np.ma.masked_array(scidata, m)         
#---get rid of cosmic rays---------------------------------------------------------
    scidata = sigma_clip(scidata, sigma=7)
#---background box-----------------------------------------------------------------
#     backbox=scidata[xybox[2]-100:xybox[2]-50, :]
#     backbox=scidata[xybox[3]+50:xybox[3]+100, :] #pulling bg in form scidata[ymin, ymax, xmin, xmax]
   
#     bkgd = backbox.mean(axis=0)
#     print('background',bkgd)
#     bkgd = sp.signal.medfilt(bkgd,31)
#     bkgd = np.array([bkgd,]*522)
    bkgd = hist_med #set this as the median from the histogram for bkgrd
    
    scidata = scidata-bkgd
    print('scidata - bkgd', scidata)
#     scidata = sigma_clip(scidata, sigma=5)
#     print('scidata shape:', np.shape(scidata))
#     print('images shape:', np.shape(images))
#     print('images_section shape:', np.shape(images[:,:,j,i]))
    images[:,:,i]=scidata


    return scidata, images

### Finds 1st order 
def getbox(scidata):
    box1 = [100,180,300,400,400,300,512,400]
#     box1 = [65, 100,170,193,250,194,500,440]
    tried = 1
    holdy=np.zeros(10)
    holdx=np.zeros(10)
    for xx in range(box1[0],box1[1],10): #CHANGE the 80 & 180 to span pixels that are on 
                                #either side of the left edge of your spectra 
                                #without encompassing contamination sources
        for yy in range(box1[2],box1[3]): #CHANGE the 0 & 250 to span pixels that are on
                                #either side of the bottom edge of your spectra
                                #without encompassing contamination sources
            ybot=yy
            if scidata[yy,xx] > 4*np.mean(scidata):
                break
        holdy[int((xx-box1[0])/10-1)]=ybot #CHANGE 80 to min value chosen for xx
    ybot=int(np.median(holdy))
    
    for xx in range(box1[0],box1[1],10): #CHANGE the 80 & 180 to span pixels that are on 
                                #either side of the left edge of your spectra 
                                #without encompassing contamination sources
        for yy in range(box1[4],box1[5], -1): #CHANGE the 450 & 0 to span pixels that are 
                                #on either side of the top edge of your spectra
                                #without encompassing contamination sources
            ytop=yy
            if scidata[yy,xx] > 2*np.mean(scidata):
                break
        holdy[int((xx-box1[0])/10-1)]=ytop #CHANGE 80 to min value chosen for xx
    ytop=int(np.median(holdy))

    for yy in range(ybot,ytop, tried):
        for xx in range(box1[0],box1[1]): #CHANGE the 0 & 350 to span pixels that are on 
                                #either side of the left edge of your spectra 
                                #without encompassing contamination sources
            xleft=xx
            if scidata[yy,xx] > 2*np.mean(scidata):
                break
        holdx[int((yy-ybot)/(tried)-1)]=xleft
    xleft=int(np.median(holdx))
    for yy in range(ybot,ytop, tried):
        for xx in range(box1[6],box1[7], -1): #CHANGE the 250 & 0 to span pixels that are on 
                                #either side of the right edge of your spectra 
                                #without encompassing contamination sources
            xright=xx
            if scidata[yy,xx] > 2*np.mean(scidata):
                break
        holdx[int((yy-ybot)/(1)-tried)]=xright
    xright=int(np.median(holdx))
    global xybox
    xybox=np.array([xleft, xright, ybot, ytop])

    print('xybox(xleft, xright, ybot, ytop)=', xybox)
    return xybox

