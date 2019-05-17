import numpy as np
import stereo

#============================================================================
#============================================================================
def origin(xx,yy,xi,yi,xf,yf):
   """
      Internal function to provide the origin of xx,yy, given that they have 
      been transformed (translated and rotated) through a transformation that
      transform the segment xi[:],yi[:] into xf[:],yf[:].

      Usage: 
        [x2d,y2d] = origin(xx,yy,xi,yi,xf,yf)

      Input:
        xx, yy : 1d numpy arrays.
        xi, yi, xf,yf : 2-element numpy arrays.

   """

   trans = [xf[0]-xi[0],yf[0]-yi[0]]
   sss = np.sqrt((xi[1]-xi[0])**2+(yi[1]-yi[0])**2)
   xx1 = (xi[1]-xi[0])/sss
   yy1 = (yi[1]-yi[0])/sss
   xx2 = (xf[1]-xf[0])/sss
   yy2 = (yf[1]-yf[0])/sss
   rotat = np.array([ [ xx1*xx2+yy1*yy2 , xx2*yy1-xx1*yy2 ], [ xx1*yy2-xx2*yy1, xx1*xx2+yy1*yy2 ] ])

   [x2d,y2d] = np.meshgrid(xx,yy)

   # coordinates of points of origin for each point of the new grid (inverse transformation) :
   x2d_origin = np.zeros(np.shape(x2d))-1.e20
   y2d_origin = np.zeros(np.shape(y2d))-1.e20
   for kk in np.arange(0,np.shape(x2d)[1],1):
     [x2d_origin[:,kk],y2d_origin[:,kk]] = np.matmul(np.linalg.inv(rotat),[x2d[:,kk]-xf[0],y2d[:,kk]-yf[0]]) \
                                           + np.swapaxes( [[xf[0]-trans[0],yf[0]-trans[1]]], 0, 1) * np.ones((1,np.size(x2d[:,kk])))

   return [x2d_origin,y2d_origin]

#============================================================================
#============================================================================
def remap(x,y,M,orientation='portrait'):
   """
     Remap Antarctic stereographic projection to maximize ice shelf sizes.

     Usage: 
       import iceshelfmap as isf
       [xnew,ynew,lonnew,latnew,msk,Mnew]=isf.remap(x,y,M) 

     Input:
       x,y are the stereographic coordinates in m [1d numpy array]
       M is the variable to remap [2d numpy array]
       orientation = 'portrait' [default], 'landscape', 'dev' (no rotation).

     Output:
       xnew,ynew = new discontinued coordinates (m) [1d numpy array]
       lonnew,latnew = (lon,lat) on new grid [2d numpy array]
       msk = mask defining the 4 areas [2d numpy array]
       Mnew = remapped 2d variable [2d numpy array]

     History:
       05/2019 : First version (Nicolas Jourdain, IGE-CNRS)

   """

   if ( x[1] < x[0] ):
     x = x[::-1]
     M = M[:,::-1]
   if ( y[1] < y[0] ):
     y = y[::-1]
     M = M[::-1,:]

   #-------------------------------------------------
   # Global transformations (applied after all the parts
   # have been concatenated together) : 
 
   # angle for global rotation of the figure
   if ( orientation == 'portrait' ) | ( orientation == 'landscape' ):
     angle_glob=-10.e0
   elif ( orientation == 'dev' ):
     angle_glob=0.e0
   else:
     print '~!@#$%^& Wrong orientation argument ==> setting to portrait.' 
     angle_glob=-10.e0
   angle_glob = np.radians(angle_glob)

   # global translation of the figure
   trans_glob = np.array([  1060.e3, 65.e3 ])  # in meters
   if ( orientation == 'dev' ):
     trans_glob = np.array([ 0.e0, 0.e0 ])

   # new frame :
   if ( orientation == 'portrait' ):
     [ xmin_frame, xmax_frame ] = np.array([-1.50e6,1.50e6])
     [ ymin_frame, ymax_frame ] = np.array([-2.15e6,2.15e6])
   elif ( orientation == 'landscape' ):
     [ xmin_frame, xmax_frame ] = np.array([-2.15e6,2.15e6])
     [ ymin_frame, ymax_frame ] = np.array([-1.50e6,1.50e6])
   else:
     [ xmin_frame, xmax_frame ] = np.array([-3.e6,3.e6])
     [ ymin_frame, ymax_frame ] = np.array([-3.e6,3.e6])

   #-------------------------------------------------
   [lon,lat] = stereo.xy_to_lonlat(x,y)
   msk = np.zeros(lon.shape)

   print np.min(lon), np.max(lon)
   print np.min(lat), np.max(lat)

   #-------------------------------------------------
   # Define masks (one value per sector):
   
   # West Ant. incl. FRIS and ROSS :
   msk[ np.where(  ( (lon <  -25.0) & (lon >=  -58.0) & (lat < (15.5*lon+77.5*58.0-25.0*62.0)/(-33.0) )   & (lat >= -90.0) ) | \
                   ( (lon <  -58.0) & (lon >=  -60.0) & (lat < -62.0)                                     & (lat >= -90.0) ) | \
                   ( (lon <  -60.0) & (lon >= -100.0) & (lat < (-8.0*lon+62.0*100.0-60.0*70.0)/(-40.0) )  & (lat >= -90.0) ) | \
                   ( (lon < -100.0) & (lon >= -140.0) & (lat < -70.0)                                     & (lat >= -90.0) ) | \
                   ( (lon < -140.0) & (lon >= -180.0) & (lat < -72.0)                                     & (lat >= -90.0) ) | \
                   ( (lon <= 180.0) & (lon >=  158.0) & (lat < -72.0)                                     & (lat >= -90.0) ) ) \
      ]=1
   # DML from Brunt to Shirase :
   msk[ np.where( ( (lon >= -30.0) & (lon < -25.0) & (lat < (4.5*lon-30.0*68.5)/30.0 ) & (lat >= (15.5*lon+77.5*58.0-25.0*62.0)/(-33.0) ) ) | \
                  ( (lon >= -25.0) & (lon < -15.0) & (lat < (4.5*lon-30.0*68.5)/30.0 ) & (lat >= -77.5) ) | \
                  ( (lon >= -15.0) & (lon <   0.0) & (lat < (4.5*lon-30.0*68.5)/30.0 ) & (lat >= (5.0*lon-15.0*72.5-20.0*77.5)/35.0 ) ) | \
                  ( (lon >=   0.0) & (lon <  20.0) & (lat < -68.5)                     & (lat >= (5.0*lon-15.0*72.5-20.0*77.5)/35.0 ) ) | \
                  ( (lon >=  20.0) & (lon <  42.0) & (lat < -68.5)                     & (lat >= (0.5*lon-42.0*72.5+20.0*72.0)/22.0 ) ) ) \
      ]=2
   # Amery, from Shirase to Publications :
   msk[ np.where( ( (lon >= 42.0) & (lon < 54.0) & (lat < -65.2 )                                 & (lat >= (-2.5*lon-68.5*60.0+42.0*71.0)/18.0 ) ) | \
                  ( (lon >= 54.0) & (lon < 60.0) & (lat < (2.0*lon-67.2*54.0+80.0*65.2)/(-26.0) ) & (lat >= (-2.5*lon-68.5*60.0+42.0*71.0)/18.0 ) ) | \
                  ( (lon >= 60.0) & (lon < 67.0) & (lat < (2.0*lon-67.2*54.0+80.0*65.2)/(-26.0) ) & (lat >= (-3.0*lon-71.0*67.0+60.0*74.0)/7.0 )  ) | \
                  ( (lon >= 67.0) & (lon < 80.0) & (lat < (2.0*lon-67.2*54.0+80.0*65.2)/(-26.0) ) & (lat >= (4.5*lon-74.0*80.0+67.0*69.5)/13.0 )  ) ) \
      ]=3
   # East Antarctica (Pacific-Indian sector)
   msk[ np.where( ( (lon >=  80.0) & (lon < 135.0) & (lat < -64.5)                                    & (lat >= -68.5) ) | \
                  ( (lon >= 135.0) & (lon < 158.0) & (lat < (5.5*lon-70.0*135.0+172.0*64.5)/(-37.0) ) & (lat >= (-3.5*lon-68.5*158.0+135.0*72.0)/23.0 ) ) | \
                  ( (lon >= 158.0) & (lon < 172.0) & (lat < (5.5*lon-70.0*135.0+172.0*64.5)/(-37.0) ) & (lat >= -72.0) ) ) \
      ]=4
  
   # Define halo so that origins out of current frame end up with mask = 0 after the transformation:
   msk[ 0,:] = 0
   msk[-1,:] = 0
   msk[:, 0] = 0
   msk[:,-1] = 0
 
   #-------------------------------------------------
   # Translations and rotations :
   
   ii_origin, jj_origin = np.meshgrid( np.arange(0,np.size(x),1), np.arange(0,np.size(y),1) )
   
   msk2 = msk * 0
   lon2 = lon * 0.e0
   lat2 = lat * 0.e0
   M2   = M   * 0.e0  
 
   # Displacements defined by 2 initial points (lon_ini_1,lat_ini_1) and (lon_ini_2,lat_ini_2)
   # and two final points (lon_fin_1,lat_fin_1) and (lon_fin_2,lat_fin_2) for each sector:
   
   #-----------
   # West Ant. incl. FRIS and ROSS :
   xi = np.array([  0.e0, -1.e6 ]) #
   yi = np.array([  0.e0,  1.e6 ]) # no transformation for this sector
   xf = xi[:]*1.e0                 # before global rotation
   yf = yi[:]*1.e0                 #

   # final segment after global transformation :
   xf[:] = xf[:] + trans_glob[0]
   yf[:] = yf[:] + trans_glob[1]
   xfnew = xf[:]*1.e0
   yfnew = yf[:]*1.e0
   xfnew[0] = np.linalg.norm([xf[0],yf[0]]) * np.cos( np.arctan2(yf[0],xf[0]) + angle_glob )
   yfnew[0] = np.linalg.norm([xf[0],yf[0]]) * np.sin( np.arctan2(yf[0],xf[0]) + angle_glob )
   xfnew[1] = np.linalg.norm([xf[1],yf[1]]) * np.cos( np.arctan2(yf[1],xf[1]) + angle_glob )
   yfnew[1] = np.linalg.norm([xf[1],yf[1]]) * np.sin( np.arctan2(yf[1],xf[1]) + angle_glob )

   # coordinates of origin for each grid point (x,y) : 
   [x2d_ori,y2d_ori]=origin(x,y,xi,yi,xfnew,yfnew)
   
   for kj in np.arange(0,np.size(y),1):
     for ki in np.arange(0,np.size(x),1):
       itmp=np.argmin((x2d_ori[kj,ki]-x)**2)
       jtmp=np.argmin((y2d_ori[kj,ki]-y)**2)
       if( msk[jtmp,itmp] == 1 ):
         ii_origin[kj,ki] = itmp
         jj_origin[kj,ki] = jtmp
         msk2[kj,ki] = 1
         lon2[kj,ki] = lon[jtmp,itmp]
         lat2[kj,ki] = lat[jtmp,itmp]
         M2  [kj,ki] = M  [jtmp,itmp]
 
   #-----------
   # DML from Brunt to Shirase :
   xi = np.array([  -560., 1231. ])*1.e3 # segments defined with no
   yi = np.array([  1540., 1467. ])*1.e3 # global transformation
   xf = np.array([ -1821., -284. ])*1.e3 # ('dev' mode)
   yf = np.array([  1587.,  716. ])*1.e3 
 
   # final segment after global transformation :
   xf[:] = xf[:] + trans_glob[0]
   yf[:] = yf[:] + trans_glob[1]
   xfnew = xf
   yfnew = yf
   xfnew[0] = np.linalg.norm([xf[0],yf[0]]) * np.cos( np.arctan2(yf[0],xf[0]) + angle_glob )
   yfnew[0] = np.linalg.norm([xf[0],yf[0]]) * np.sin( np.arctan2(yf[0],xf[0]) + angle_glob )
   xfnew[1] = np.linalg.norm([xf[1],yf[1]]) * np.cos( np.arctan2(yf[1],xf[1]) + angle_glob )
   yfnew[1] = np.linalg.norm([xf[1],yf[1]]) * np.sin( np.arctan2(yf[1],xf[1]) + angle_glob )

   # coordinates of origin for each grid point (x,y) : 
   [x2d_ori,y2d_ori]=origin(x,y,xi,yi,xfnew,yfnew)
   
   for kj in np.arange(0,np.size(y),1):
     for ki in np.arange(0,np.size(x),1):
       itmp=np.argmin((x2d_ori[kj,ki]-x)**2)
       jtmp=np.argmin((y2d_ori[kj,ki]-y)**2)
       if( msk[jtmp,itmp] == 2 ):
         ii_origin[kj,ki] = itmp
         jj_origin[kj,ki] = jtmp
         msk2[kj,ki] = 2
         lon2[kj,ki] = lon[jtmp,itmp]
         lat2[kj,ki] = lat[jtmp,itmp]
         M2  [kj,ki] = M  [jtmp,itmp]
 
   #-----------
   # Amery, from Shirase to Publications :
   xi = np.array([  2161.,  1411. ])*1.e3 # segments defined with no
   yi = np.array([   381.,  1681. ])*1.e3 # global transformation
   xf = np.array([ -1435.,  -441. ])*1.e3 # ('dev' mode)
   yf = np.array([  -656.,   700. ])*1.e3
   
   # final segment after global transformation :
   xf[:] = xf[:] + trans_glob[0]
   yf[:] = yf[:] + trans_glob[1]
   xfnew = xf
   yfnew = yf
   xfnew[0] = np.linalg.norm([xf[0],yf[0]]) * np.cos( np.arctan2(yf[0],xf[0]) + angle_glob )
   yfnew[0] = np.linalg.norm([xf[0],yf[0]]) * np.sin( np.arctan2(yf[0],xf[0]) + angle_glob )
   xfnew[1] = np.linalg.norm([xf[1],yf[1]]) * np.cos( np.arctan2(yf[1],xf[1]) + angle_glob )
   yfnew[1] = np.linalg.norm([xf[1],yf[1]]) * np.sin( np.arctan2(yf[1],xf[1]) + angle_glob )

   # coordinates of origin for each grid point (x,y) : 
   [x2d_ori,y2d_ori]=origin(x,y,xi,yi,xfnew,yfnew)
   
   for kj in np.arange(0,np.size(y),1):
     for ki in np.arange(0,np.size(x),1):
       itmp=np.argmin((x2d_ori[kj,ki]-x)**2)
       jtmp=np.argmin((y2d_ori[kj,ki]-y)**2)
       if( msk[jtmp,itmp] == 3 ):
         ii_origin[kj,ki] = itmp
         jj_origin[kj,ki] = jtmp
         msk2[kj,ki] = 3
         lon2[kj,ki] = lon[jtmp,itmp]
         lat2[kj,ki] = lat[jtmp,itmp]
         M2  [kj,ki] = M  [jtmp,itmp]
   
   #-----------
   # East Antarctica (Pacific-Indian sector) :
   xi = np.array([  2062.,   751. ])*1.e3 # segments defined with no
   yi = np.array([  -751., -2062. ])*1.e3 # global transformation
   xf = np.array([ -1003., -2308. ])*1.e3 # ('dev' mode)
   yf = np.array([ -1271.,    39. ])*1.e3
   
   # final segment after global transformation :
   xf[:] = xf[:] + trans_glob[0]
   yf[:] = yf[:] + trans_glob[1]
   xfnew = xf
   yfnew = yf
   xfnew[0] = np.linalg.norm([xf[0],yf[0]]) * np.cos( np.arctan2(yf[0],xf[0]) + angle_glob )
   yfnew[0] = np.linalg.norm([xf[0],yf[0]]) * np.sin( np.arctan2(yf[0],xf[0]) + angle_glob )
   xfnew[1] = np.linalg.norm([xf[1],yf[1]]) * np.cos( np.arctan2(yf[1],xf[1]) + angle_glob )
   yfnew[1] = np.linalg.norm([xf[1],yf[1]]) * np.sin( np.arctan2(yf[1],xf[1]) + angle_glob )

   # coordinates of origin for each grid point (x,y) : 
   [x2d_ori,y2d_ori]=origin(x,y,xi,yi,xfnew,yfnew)
   
   for kj in np.arange(0,np.size(y),1):
     for ki in np.arange(0,np.size(x),1):
       itmp=np.argmin((x2d_ori[kj,ki]-x)**2)
       jtmp=np.argmin((y2d_ori[kj,ki]-y)**2)
       if( msk[jtmp,itmp] == 4 ):
         ii_origin[kj,ki] = itmp
         jj_origin[kj,ki] = jtmp
         msk2[kj,ki] = 4
         lon2[kj,ki] = lon[jtmp,itmp]
         lat2[kj,ki] = lat[jtmp,itmp]
         M2  [kj,ki] = M  [jtmp,itmp]
   
   M2[ msk2 == 0 ] = np.nan

   #-------------------------------------------------
   if ( orientation == 'landscape' ):
     M2=np.swapaxes(M2,1,0)
     msk2=np.swapaxes(msk2,1,0)
     lon2=np.swapaxes(lon2,1,0)
     lat2=np.swapaxes(lat2,1,0)
     tmpp=x
     x=y
     y=tmpp

   #-------------------------------------------------
   # reframe output 
   imin_frame = np.argmin((x-xmin_frame)**2)
   imax_frame = np.argmin((x-xmax_frame)**2)
   jmin_frame = np.argmin((y-ymin_frame)**2)
   jmax_frame = np.argmin((y-ymax_frame)**2)
 
   #-------------------------------------------------
   # Put a halo of msk2=0 (to have contours even without frame) :

   msk2[jmin_frame,:] = 0
   msk2[jmax_frame,:] = 0
   msk2[:,imin_frame] = 0
   msk2[:,imax_frame] = 0

   #-------------------------------------------------
   return [x[imin_frame:imax_frame],y[jmin_frame:jmax_frame],lon2[jmin_frame:jmax_frame,imin_frame:imax_frame], \
                                                             lat2[jmin_frame:jmax_frame,imin_frame:imax_frame], \
                                                             msk2[jmin_frame:jmax_frame,imin_frame:imax_frame], \
                                                               M2[jmin_frame:jmax_frame,imin_frame:imax_frame] ]
