import numpy as np
import stereo

#============================================================================
#============================================================================
#def origin(xx,yy,loni,lati,lonf,latf):
def origin(xx,yy,xi,yi,xf,yf):
   """
      Internal function.

      Usage: 
        [x2d,y2d] = origin(xx,yy,loni,lati,lonf,latf)

      All inputs are 2d numpy arrays defining initial and final longitude
      and latitude of 2 points. Returns original coordinates of rotated (x,y)
   """

   #[xi,yi] = stereo.lonlat_to_xy(loni,lati)
   #[xf,yf] = stereo.lonlat_to_xy(lonf,latf)
   print [xi,yi],  [xf,yf] 

   trans = [xf[0]-xi[0],yf[0]-yi[0]]
   sss = np.sqrt((xi[1]-xi[0])**2+(yi[1]-yi[0])**2)
   xx1 = (xi[1]-xi[0])/sss
   yy1 = (yi[1]-yi[0])/sss
   xx2 = (xf[1]-xf[0])/sss
   yy2 = (yf[1]-yf[0])/sss
   rotat = np.array([ [ xx1*xx2+yy1*yy2 , xx2*yy1-xx1*yy2 ], [ xx1*yy2-xx2*yy1, xx1*xx2+yy1*yy2 ] ])
   ## Check inverse transformation:
   #[check_xi, check_yi] = np.matmul( np.linalg.inv(rotat),[xf[:]-xf[0],yf[:]-yf[0]] ) + np.swapaxes( [[xf[0]-trans[0],yf[0]-trans[1]]], 0, 1) * np.ones((1,np.size(xf[:])))

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
def remap(x,y,M):
   """
     Remap Antarctic stereographic projection to maximize ice shelf sizes.

     Usage: 
       import iceshelfmap as isf
       [xnew,ynew,lonnew,latnew,msk,Mnew]=isf.remap(x,y,M) 

     Input:
       x,y are the stereographic coordinates in m [1d numpy array]
       M is the variable to remap [2d numpy array]

     Output:
       xnew,ynew = new discontinued coordinates (m) [1d numpy array]
       lonnew,latnew = (lon,lat) on new grid [2d numpy array]
       msk = mask defining the 4 areas [2d numpy array]
       Mnew = remapped 2d variable [2d numpy array]

     History:
       05/2019 : First version (Nicolas Jourdain, IGE-CNRS)

   """
  
   [lon,lat] = stereo.xy_to_lonlat(x,y)
   msk = np.zeros(lon.shape)
 
   #-------------------------------------------------
   # Define masks (one value per sector):
   
   # West Ant. incl. FRIS and ROSS :
   #msk[ np.where(   ( (lon <  -40.0) & (lon >= -110.0) & (lat < -62.0) & (lat >= -85.0) ) \
   #               | ( (lon < -110.0) & (lon >= -140.0) & (lat < -73.0) & (lat >= -77.5) ) \
   #               | ( (lon < -140.0) & (lon >= -180.0) & (lat < -73.0) & (lat >= -87.0) ) \
   #               | ( (lon <  -25.0) & (lon >=  -40.0) & (lat < -77.5) & (lat >= -85.0) ) \
   #               | ( (lon <= 180.0) & (lon >=  158.0) & (lat < -72.0) & (lat >= -87.0) ) ) \
   #   ]=1
   msk[ np.where(  ( (lon <  -40.0) & (lon >= -110.0) & (lat < -62.0) & (lat >= -90.0) ) | \
                   ( (lon < -110.0) & (lon >= -180.0) & (lat < (-10.0*lon+62.0*202.0-110.0*72.0)/(-92.0) ) & (lat >= -90.0) ) | \
                   ( (lon <= 180.0) & (lon >=  158.0) & (lat < (-10.0*lon-62.0*158.0+250.0*72.0)/(-92.0) ) & (lat >= -90.0) ) | \
                   ( (lon <  -25.0) & (lon >=  -40.0) & (lat < (15.5*lon+40.0*77.5-25.0*62.0)/(-15.0) )    & (lat >= -90.0) ) ) \
      ]=1
   # DML from Brunt to Shirase :
   #msk[ np.where( ( (lon >= -30.0) & (lon < -15.0) & (lat < -68.5) & (lat >= -77.5) ) | \
   #               ( (lon >= -15.0) & (lon <   0.0) & (lat < -68.5) & (lat >= -75.0) ) | \
   #               ( (lon >=   0.0) & (lon <  42.0) & (lat < -68.5) & (lat >= -72.5) ) ) \
   #   ]=2
   msk[ np.where( ( (lon >= -30.0) & (lon < -15.0) & (lat < -68.5) & (lat >= -77.5) ) | \
                  ( (lon >= -15.0) & (lon <  20.0) & (lat < -68.5) & (lat >= (5.0*lon-15.0*72.5-20.0*77.5)/35.0 ) ) | \
                  ( (lon >=  20.0) & (lon <  42.0) & (lat < -68.5) & (lat >= (0.5*lon-42.0*72.5+20.0*72.0)/22.0 ) ) ) \
      ]=2
   # Amery, from Shirase to Publications :
   msk[ np.where( ( (lon >= 42.0) & (lon < 65.0) & (lat < -65.5) & (lat >= -69.0) ) | \
                  ( (lon >= 65.0) & (lon < 75.0) & (lat < -67.0) & (lat >= -74.0) ) | \
                  ( (lon >= 75.0) & (lon < 80.0) & (lat < -67.0) & (lat >= -71.0) ) ) \
      ]=3
   # East Antarctica (Pacific-Indian sector)
   #msk[ np.where( ( (lon >=  80.0) & (lon < 135.0) & (lat < -64.5) & (lat >= -68.0) ) | \
   #               ( (lon >= 135.0) & (lon < 158.0) & (lat < -64.5) & (lat >= -70.0) ) | \
   #               ( (lon >= 158.0) & (lon < 172.0) & (lat < -64.5) & (lat >= -72.0) ) ) \
   #   ]=4
   msk[ np.where( ( (lon >=  80.0) & (lon < 135.0) & (lat < -64.5) & (lat >= -68.5) ) | \
                  ( (lon >= 135.0) & (lon < 158.0) & (lat < -64.5) & (lat >= (-3.5*lon-68.5*158.0+135.0*72.0)/23.0 ) ) | \
                  ( (lon >= 158.0) & (lon < 172.0) & (lat < -64.5) & (lat >= -72.0) ) ) \
      ]=4
   
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
   msk2[ msk==1 ] = 1
   lon2[ msk==1 ] = lon[ msk==1 ]
   lat2[ msk==1 ] = lat[ msk==1 ]
   M2  [ msk==1 ] = M  [ msk==1 ]
   ii_origin[ msk!=1 ]=-999999
   jj_origin[ msk!=1 ]=-999999
   
   #-----------
   # DML from Brunt to Shirase :
   xi = np.array([  -560., 1231. ])*1.e3
   yi = np.array([  1540., 1467. ])*1.e3
   xf = np.array([ -1771., -234. ])*1.e3
   yf = np.array([  1587.,  716. ])*1.e3 
   aa=np.sqrt((xi[1]-xi[0])**2+(yi[1]-yi[0])**2)
   bb=np.sqrt((xf[1]-xf[0])**2+(yf[1]-yf[0])**2)
   print [xi+(xf-xi)*aa/bb], [yi+(yf-yi)*aa/bb]
   #loni = [-20.0, 40.0] # longitude of two initial points
   #lati = [-75.0,-72.5] # latitude of two initial points
   #lonf = [-50.0,  0.0] # longitude of two final points
   #latf = [-71.0,-85.0] # latitude of two final points
   
   #[x2d_ori,y2d_ori]=origin(x,y,loni,lati,lonf,latf)
   [x2d_ori,y2d_ori]=origin(x,y,xi,yi,xf,yf)
  
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
   xi = np.array([  2161.,  1411. ])*1.e3
   yi = np.array([   381.,  1681. ])*1.e3
   xf = np.array([ -1435.,  -441. ])*1.e3
   yf = np.array([  -656.,   700. ])*1.e3
   aa=np.sqrt((xi[1]-xi[0])**2+(yi[1]-yi[0])**2)
   bb=np.sqrt((xf[1]-xf[0])**2+(yf[1]-yf[0])**2)
   print [xi+(xf-xi)*aa/bb], [yi+(yf-yi)*aa/bb]
   #loni = [  80.0,  40.0] # longitude of two initial points
   #lati = [ -70.0, -70.0] # latitude of two initial points
   #lonf = [ 120.0, -20.0] # longitude of two final points
   #latf = [ -77.5, -85.5] # latitude of two final points
   
   #[x2d_ori,y2d_ori]=origin(x,y,loni,lati,lonf,latf)
   [x2d_ori,y2d_ori]=origin(x,y,xi,yi,xf,yf)
   
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
   xi = np.array([  2062.,   751. ])*1.e3
   yi = np.array([  -751., -2062. ])*1.e3
   xf = np.array([  -973., -2278. ])*1.e3
   yf = np.array([ -1371.,   -61. ])*1.e3
   aa=np.sqrt((xi[1]-xi[0])**2+(yi[1]-yi[0])**2)
   bb=np.sqrt((xf[1]-xf[0])**2+(yf[1]-yf[0])**2)
   print [xi+(xf-xi)*aa/bb], [yi+(yf-yi)*aa/bb]
   #loni = [ 110.0, 160.0] # longitude of two initial points
   #lati = [ -70.0, -70.0] # latitude of two initial points
   #lonf = [-155.0, -87.0] # longitude of two final points
   #latf = [ -74.5, -70.5] # latitude of two final points
   
   #[x2d_ori,y2d_ori]=origin(x,y,loni,lati,lonf,latf)
   [x2d_ori,y2d_ori]=origin(x,y,xi,yi,xf,yf)
   
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
   
   #-------------------------------------------------
   M2[ msk2 == 0 ] = np.nan

   return [x,y,lon2,lat2,msk2,M2]
