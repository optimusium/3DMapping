# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 08:21:04 2020

@author: boonping
"""

import os,re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
import math
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull,Delaunay
from scipy import ndimage
from scipy.misc import imsave
from scipy.misc import imread
from PIL import Image
from skimage.draw import line_aa


from tensorflow.keras.models import Model,load_model
from keras.preprocessing.image import save_img
from tensorflow.keras import backend

#load weight and model files
model = load_model('map_network_resnet_classifier.hdf5')
model.load_weights("map_network_resnet_weight.hdf5")

from skimage.util import view_as_windows as viewW

#slice size set to 1024 for memory limitation
slicesize=1024
#display debug detail
display=1

#opening scan file sets in sequence
for kk in range(0,10,1):
    #opening pose file #GPS/IMU data
    inf0=open("scan%03i.pose" % kk,"r")
    pos0=inf0.readline().replace("\n","").replace(" ",",")
    pos0=np.array(eval("["+pos0+"]"))
    eu=inf0.readline().replace("\n","").replace(" ",",")
    print("913",eu)
    eu=np.array( eval("["+eu+"]")  )*math.pi/180
    inf0.close()

    #generate rotation matrix (processing .pose file)
    matz=[[ math.cos(eu[2]) , -1*math.sin(eu[2]) , 0 ],[ math.sin(eu[2]) ,  1*math.cos(eu[2]) , 0 ],[ 0 , 0, 1] ]
    maty=[[ math.cos(eu[1]) , 0, 1*math.sin(eu[1]) ],[ 0 , 1, 0],[ -1*math.sin(eu[1]) , 0, 1*math.cos(eu[1])] ]
    matx=[ [1,0,0],[ 0, math.cos(eu[0]) , -1*math.sin(eu[0]) ],[ 0, math.sin(eu[0]) ,  1*math.cos(eu[0]) ] ]
    matz=np.array(matz)
    maty=np.array(maty)
    matx=np.array(matx)

    mat=np.matmul(matz,maty)
    mat=np.matmul(mat,matx)

    #process scan file 
    inf=open("scan%03i.3d" % kk,"r")
    buf=[]
    buf2=[]
    #Getting position from line in text file
    pos=inf.readline().replace("\n","").replace(" ",",")
    pos=np.array(eval("["+pos+"]"))

    poolmaxx=-100000
    poolminx=100000
    poolmaxy=-100000
    poolminy=100000
    #scan1 will be used to store all the scan points
    scan1=np.array([])

    #Fitting height value. 
    for line in inf.readlines():
        line=line.replace("\n","").replace(" ",",")
        #print(line)
        p=eval("["+line+"]")
        buf.append(eval("["+line+"]"))
        
        posi=np.transpose( np.array(eval("["+line+"]")) )
        #res=np.round( np.matmul(mat,posi) ).astype('int')
        res=np.round( np.matmul(mat,posi)+pos0 ).astype('int')
    
        #Taking out noise which gives negative value. 
        if res[1]<-5: continue
        if res[1]<0: res[1]=0
        #Inverting value (255 is the maximum height)
        res[1]=255-res[1]
        #Capping the maximum height to be 192 pixel
        if res[1]<64: res[1]=64
        #if res[1]<128: res[1]=128

        #All scan points are in scan1
        scan1=np.append(scan1,res)
        scan1=scan1.reshape(int(scan1.shape[0]/3),3)

        ###########################################################################################
        #IF there are multiple height value for a 2D grid point, choose based on following criteria
        # taking value in between 245 to 251. (obstacle in these height are targeted)
        #If value is larger than 251, it is taken as maneuverable ground value.
        if scan1.shape[0]==0:
            scan1=np.append(scan1,res)
            scan1=scan1.reshape(int(scan1.shape[0]/3),3)
        elif res[1]>=245 and res[1]<251:
            if scan1[ (scan1[:,0]==res[0]) & (scan1[:,2]==res[2]) ].shape[0]>0:
                scan1[ (scan1[:,0]==res[0]) & (scan1[:,2]==res[2]) ][:,1]=res[1]
            else:
                scan1=np.append(scan1,res)
                #print(scan1,res,scan1.shape)
                scan1=scan1.reshape(int(scan1.shape[0]/3),3)
        elif res[1]>=251:
            tempa=scan1[ (scan1[:,0]==res[0]) & (scan1[:,2]==res[2]) ]
            if tempa.shape[0]>0:
                if np.max(tempa[:,1])>246 and np.max(tempa[:,1])<251 :
                    continue
                scan1[ (scan1[:,0]==res[0]) & (scan1[:,2]==res[2]) ][:,1]=res[1]
            else:
                scan1=np.append(scan1,res)
                #print(scan1,res,scan1.shape)
                scan1=scan1.reshape(int(scan1.shape[0]/3),3)
                
        elif scan1[ (scan1[:,0]==res[0]) & (scan1[:,2]==res[2]) ].shape[0]>0:
            if np.min(scan1[ (scan1[:,0]==res[0]) & (scan1[:,2]==res[2]) ][:,1])>=245:
                continue
            if res[1]>np.max( scan1[ (scan1[:,0]==res[0]) & (scan1[:,2]==res[2]) ][:,1] )  and np.min(scan1[ (scan1[:,0]==res[0]) & (scan1[:,2]==res[2]) ][:,1])>=64:
                scan1[ (scan1[:,0]==res[0]) & (scan1[:,2]==res[2]) ][:,1]=res[1]

        else:
            scan1=np.append(scan1,res)
            scan1=scan1.reshape(int(scan1.shape[0]/3),3)
        ############################################################################################

    inf.close()
    #End processing scan file
                        
    #skip further processing if there is no measurement captured in the scan file
    if scan1.shape[0]==0: continue

    #sorting the scan points
    scan1= scan1[ (scan1[:,0]).argsort() ]
    
    #if scan1.shape[0]==0: continue
    print(scan1.shape)
    scan1=scan1.astype('int')
    
    #getting unique scan points per scan file 
    scan1=np.unique(scan1,axis=0)
    print(scan1.shape)
    
   
    #raise
    
    #Calulate the max and min x,y value
    if np.max(scan1[:,0])>poolmaxx: poolmaxx=np.max(scan1[:,0])
    
    if np.min(scan1[:,0])<poolminx: poolminx=np.min(scan1[:,0])
    
    if np.max(scan1[:,2])>poolmaxy: poolmaxy=np.max(scan1[:,2])
    
    if np.min(scan1[:,2])<poolminy: poolminy=np.min(scan1[:,2])


    xmin=np.min(scan1[:,0])
    xmax=np.max(scan1[:,0])
    ymin=np.min(scan1[:,2])
    ymax=np.max(scan1[:,2])
    zmin=np.min(scan1[:,1])
    zmax=np.max(scan1[:,1])
    print(xmax,xmin,ymax,ymin,zmin,zmax)
    #raise
    
    #calculate the slice number needed on x and y axis (targeted slice size is 1024x1024)
    nx=np.ceil( (xmax)/slicesize )
    ny=np.ceil( (ymax)/slicesize )
    #print(nx,ny)
    
    #Calulate the offset for each 1024x1024 slice
    startx=math.floor(xmin/slicesize)*slicesize
    starty=math.floor(ymin/slicesize)*slicesize
    endx=math.floor(xmax/slicesize)*slicesize
    endy=math.floor(ymax/slicesize)*slicesize
                        
    #setup map1 as overall 2D map
    x=np.arange(slicesize)
    y=np.arange(slicesize)
    map1=np.zeros((endx+slicesize-startx,endy+slicesize-starty))

    #create maps to interpolate the points
    pos1=np.copy(pos0)
    pos1[0]-=startx
    pos1[2]-=starty
    pos1=pos1.astype('int')
    #Converting all scan points to points in 2D map
    map1[scan1[:,0]-startx,scan1[:,2]-starty]=scan1[:,1]
    print("done extract map1",map1.shape)
    
    #---interpolation of scan point to sensor------------------------------------------
    #getting x,y,z 
    xx0=np.copy(scan1[:,0])
    yy0=np.copy(scan1[:,2])
    zz0=np.copy(scan1[:,1])
    poi=np.dstack([xx0,yy0])
    poi=poi.reshape(poi.shape[1],poi.shape[2])
    #print(poi)
    #getting all the angles
    an=np.arctan((xx0-pos0[0])/((yy0-pos0[2]+0.0001)))*180/3.1416
    #print(an.argsort())
    poi= poi[ an.argsort() ]
    zz0= zz0[ an.argsort() ]
    #print(an)
    #print(poi)
    
    #using distance at x,y,z to calculate angle
    dif=np.abs(np.diff(poi,axis=0) )
    difxy=dif[:,0]+dif[:,1]
    difxy[difxy<2]=33
    difz=np.abs( np.diff(zz0) )
    
    #print(difxy,difz)
    poi1=np.copy(poi[:-1])
    poi1=poi1[(difxy<64)&(difz<10)]
    poi2=np.copy(poi[1:])
    poi2=poi2[(difxy<64)&(difz<10)]
    
    poi=(poi1+poi2)/2
    poi=poi.astype('int')

    poi3=poi1+((poi2-poi1)/4)
    poi3=poi3.astype('int')

    poi4=poi1+(3*(poi2-poi1)/4)
    poi4=poi4.astype('int')

    #getting interpolation value in between nearest scan point
    zz1=np.copy(zz0[:-1])
    zz1=zz1[(difxy<64)&(difz<10)]
    zz2=np.copy(zz0[1:])
    zz2=zz2[(difxy<64)&(difz<10)]
    zz0=(zz1+zz2)/2
    zz0=zz0.astype('int')

    zz3=zz1+(zz2-zz1)/4
    zz3=zz3.astype('int')
    zz4=zz1+3*(zz2-zz1)/4
    zz4=zz4.astype('int')
    
    #------------------------------
    #Getting interpolation in between nearest scan point into map1
    map1[poi[:,0]-startx,poi[:,1]-starty]=zz0    
    map1[poi3[:,0]-startx,poi3[:,1]-starty]=zz3    
    map1[poi4[:,0]-startx,poi4[:,1]-starty]=zz4    
    #------------------------------
    
   
    #use map1b to interpolate value>245
    #use map1c to interpolate value<245 (value<64 are non-confidence value)
    map1c=np.copy(map1)
    map1c[map1c<64]=0
    map1c[map1c>245]=0
    map1b=np.copy(map1)
    map1b[map1b<246]=0
    
    
    xx,yy=np.nonzero(map1c)

    #interpolate the value on map1c first using scipy function line_aa
    
    #interpolate higher value first. (These extreme value can be overriden by lower value since it can be outlier)
    for i in range(xx.shape[0]):
        rr, cc, val = line_aa(xx[i], yy[i], pos1[0], pos1[2])
        dist=(((xx[i]-pos1[0])**2 )+((yy[i]-pos1[2])**2 ))**0.5
        dis=(((rr-xx[i])**2 )+((cc-yy[i])**2 ))**0.5
        val=map1c[xx[i],yy[i]]+((253-map1c[xx[i],yy[i]])*dis/dist)
        #val[val<64]=64
        val=val.astype('int')
        rr=rr[val==64]
        cc=cc[val==64]
        val=val[val==64]
        
        #print(rr,cc,val,map1c[xx[i],yy[i]])
        #raise
        map1[rr, cc] = 64
    
    #interpolate other height value
    for i in range(xx.shape[0]):
        rr, cc, val = line_aa(xx[i], yy[i], pos1[0], pos1[2])
        dist=(((xx[i]-pos1[0])**2 )+((yy[i]-pos1[2])**2 ))**0.5
        dis=(((rr-xx[i])**2 )+((cc-yy[i])**2 ))**0.5
        #val=map1c[xx[i],yy[i]]+((253-map1c[xx[i],yy[i]])*dis/dist)
        val=map1c[xx[i],yy[i]]+((253-map1c[xx[i],yy[i]])*dis/dist)
        #val[val<64]=64
        val=val.astype('int')
        rr=rr[val>244]
        cc=cc[val>244]
        val=val[val>244]
        #print(rr,cc,val,map1c[xx[i],yy[i]])
        #raise
        map1[rr, cc] = val
    
    
    #interpolate the value on map1b using scipy function line_aa
    xx,yy=np.nonzero(map1b)
    #print(xx,yy)
    #print(pos1)

    #interpolate low ground value last since it can consists of ground value. (note that height value is inverted here so that MaxPooling2D value can recognize the value)
    for i in range(xx.shape[0]):
        rr, cc, val = line_aa(xx[i], yy[i], pos1[0], pos1[2])
        dist=(((xx[i]-pos1[0])**2 )+((yy[i]-pos1[2])**2 ))**0.5
        dis=(((rr-xx[i])**2 )+((cc-yy[i])**2 ))**0.5
        val=map1b[xx[i],yy[i]]+((253-map1b[xx[i],yy[i]])*dis/dist)
        #val[val<64]=64
        val=val.astype('int')
        rr=rr[val>244]
        cc=cc[val>244]
        val=val[val>244]
        #if xx[i]<2048 and yy[i]<2048:
            #print(rr,cc,val,map1b[xx[i],yy[i]])
            #print("958a",xx[i],yy[i],rr,cc)
        #raise
        map1[rr, cc] = 253

    #raise
    
    for ii in range(0, map1.shape[0], slicesize):
        for jj in range(0, map1.shape[1], slicesize):
            if ii<poolminx-startx or ii>=poolmaxx-startx+slicesize: continue
            if jj<poolminy-starty or jj>=poolmaxy-starty+slicesize: continue
            # map2 is generated from map1 which is the overall 2D map
            map2=map1[ii:ii+slicesize,jj:jj+slicesize]
            
            #disallow slice with less than 5 points to continue processing.
            if map2[map2>0].shape[0]<5 and map2[map2>244].shape[0]==0: continue
            
            #Concatenate with previous input map that consists of points from previous scan file.
            if os.path.exists("mapim%i_%i.png" %(ii+startx,jj+starty)):
                maptemp=imread( "mapim%i_%i.png" %(ii+startx,jj+starty) )
                map2[maptemp>map2]=maptemp[maptemp>map2]

            #save input sliced image
            imsave("mapim%i_%i.png" %(ii+startx,jj+starty), map2)

            
            
            ###############################################################################
            #backup map2 as map5. Map5 will now be a copy of sliced input map that will be used as comprison against predicted output.
            map5=np.copy(map2)
            #remove salt noise by applying maximum filter over small area of pixel.
            map6=np.copy(map5)
            
            for rep in range(5):
                map6=ndimage.maximum_filter(map6, size=2)            
            map7=np.copy(map6)
            for rep in range(18):
                map6=ndimage.maximum_filter(map6, size=3)

            map2[(map7>map2)&(map7[:,0]<6)]=map7[(map7>map2)&(map7[:,0]<6)]
            map2[(map7>map2)&(map7[:,1]<6)]=map7[(map7>map2)&(map7[:,1]<6)]
            map2[(map7>map2)&(map7[:,0]>slicesize-6)]=map7[(map7>map2)&(map7[:,0]>slicesize-6)]
            map2[(map7>map2)&(map7[:,1]>slicesize-6)]=map7[(map7>map2)&(map7[:,1]>slicesize-6)]
            ###################################################################################

            #############
            #padding the edge for edge noise removal
            map2=np.pad(map2,(32,32),"edge")
            #############
            
            #normalization
            map2=map2/255

            
            #reshape input to fit the channel
            map2=map2.reshape(map2.shape[0],map2.shape[1],1)
            #prediction using neural network
            res_model=model.predict(np.expand_dims(map2,axis=0))
            
            #reshape the output 
            im=255*res_model[0][:,:,0] #.reshape(slicesize,slicesize,1)
            #value <64 are uncertainty, assign 0 value.
            im[im<64]=0

            #############
            #cropping out the edge
            im=im[32:im.shape[0]-32 , 32:im.shape[1]-32 ]
            #############
            
            im[map5>im]=map5[map5>im]
            im[ (im>230) & (map6>im) ]=map6[(im>230) & (map6>im)]
            im=im.astype('int')
            #im=fill_zero_regions(im.astype('int'))
            #save the output image in gray value for the slice
            imsave("res%i_%i.png" %(ii+startx,jj+starty), im)

            ###############################################################
            #creating Color Map from array value 
            poins1,poins3=np.nonzero(im)
            poins2=np.copy(im[np.nonzero(im)])
            #print(poins1,poins2,poins3)
            #print(np.dstack([poins1+startx,poins2.astype('int'),poins3+starty]))
            res23=np.dstack([poins1+startx+ii,poins2.astype('int'),poins3+starty+jj])
            res23=res23.reshape(res23.shape[1],res23.shape[2])
            print("928a",res23.shape)

            print("928",ii+startx,jj+starty)
            print(np.min(res23[:,0]),np.max(res23[:,0]))
            print(np.min(res23[:,1]),np.max(res23[:,1]))
            
            im2=np.zeros((slicesize,slicesize,3))
            
            #value < 64 , carries black since it is allocated as uncertainty value.
            #ima carries red color pixels. These represent ground value
            #imb carries green color pixels. Not used
            #imc carries blue color pixels. These represent obstacle value
            ima=np.copy(im)
            imb=np.copy(im)
            imc=np.copy(im)
            #print("929",ima[ima>65])
            #ima[ima<230]=0
            ima[ima<244]=0
            #ima[ima>0]=255

            imb=np.zeros(im.shape)
            
            #print("927",imb[imb>0].shape)
            
            imc[(imc<64)]=0
            imc[(imc>244)]=0
            imc[imc>0]=255-imc[imc>0]
            #print(imc[imc>0].shape)
            
            im2[:,:,0]=ima
            im2[:,:,1]=imb
            im2[:,:,2]=imc

            #create color png file based on prediction output.
            imsave("im%i_%i.png" % (ii+startx,jj+starty), im2)
            
            ###############################################################
        #getting .png file list.
        xlist=[]
        ylist=[]
        for file in os.listdir():
            if file.startswith('im') and file.count('_') and file.count('.png'):
                #print(file)
                buf=re.split("_|\.",file)
                #print(buf)
                xpos=eval(buf[0].replace("im",""))
                ypos=eval(buf[1])
                xlist.append(xpos)
                ylist.append(ypos)

        #print( max(xlist))
    
    if display==1:
        #merging color map
        new_im = Image.new('RGB', (max(xlist)-min(xlist)+1024,max(ylist)-min(ylist)+1024), (250,250,250))
        for jj in range( min(ylist), max(ylist)+1024, 1024):
            for ii in range( min(xlist), max(xlist)+1024, 1024):
            
                if os.path.exists("im%i_%i.png" % (ii,jj)):
                    img = Image.open("im%i_%i.png" % (ii,jj) )
                    print(ii,jj,img.size)
                    #new_im = Image.new('RGB', (2*img.size[0],2*img.size[1]), (250,250,250))
                    #new_im.paste(img, (ii-min(xlist),jj-min(ylist)))
                    new_im.paste(img, (jj-min(ylist),ii-min(xlist)))
                    
        new_im.save("merged_images.png", "PNG")

        #merging gray value map
        new_im = Image.new('L', (max(xlist)-min(xlist)+1024,max(ylist)-min(ylist)+1024), (250))
        for jj in range( min(ylist), max(ylist)+1024, 1024):
            for ii in range( min(xlist), max(xlist)+1024, 1024):
            
                if os.path.exists("res%i_%i.png" % (ii,jj)):
                    img = Image.open("res%i_%i.png" % (ii,jj) )
                    print(ii,jj,img.size)
                    #new_im = Image.new('RGB', (2*img.size[0],2*img.size[1]), (250,250,250))
                    #new_im.paste(img, (ii-min(xlist),jj-min(ylist)))
                    new_im.paste(img, (jj-min(ylist),ii-min(xlist)))
                    
        new_im.save("merged_gray_images.png", "PNG")
        
            
