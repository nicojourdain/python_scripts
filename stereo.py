import numpy as np
import sys

############################################################################
############################################################################
def lonlat_to_xy(lon_in,lat_in,RT=6378137.000,ex=0.08181919,lat_true=-71.000,lon_posy=0.e0):
   """
     Converts (lon,lat) to stereographic (x,y).

     Usage :

        import stereo
        x=stereo.lonlat_to_xy(lon_in,lat_in)[0]
        y=stereo.lonlat_to_xy(lon_in,lat_in)[1]

     Optional arguments (default: WGS84):
        RT : Earth radius (m)
        ex : excentricity (0-1)
        lat_true : latitude of true scale
        lon_posy : meridian of positive Y axis

     History: N. Jourdain, IGE-CNRS, April 2019 (from Andy Bliss' matlab)

   """

   #- input checking :
   if ( (RT < 6.e6) | (RT > 7.e6) ):
     sys.exit("~!@#$%^* error : Earth radius is wrong >>>>>> stop !!")
   #-
   if ( (ex < 0.0) | (ex > 1.0) ):
     sys.exit("~!@#$%^* error : Earth excentricity is wrong >>>>>> stop !!")
   #-
   if ( (lat_true < -90.0) | (lat_true > 90.0) ):
     sys.exit("~!@#$%^* error : latitude of true scale is wrong >>>>>> stop !!")
   #-
   if ( (lon_posy < -180.0) | (lon_posy > 360.0) ):
     sys.exit("~!@#$%^* error : meridian of positive Y axis is wrong >>>>>> stop !!")

   #- convert to radians :
   deg2rad = np.pi / 180.e0
   lat   = deg2rad * lat_in
   lat_c = deg2rad * lat_true
   lon   = deg2rad * lon_in
   lon_0 = deg2rad * lon_posy

   #- if the standard parallel is in S.Hemi., switch signs.
   if ( lat_c < 0.0 ):
     pm    = -1    # plus or minus, north lat. or south
     lat   = -lat
     lat_c = -lat_c
     lon   = -lon
     lon_0 = -lon_0
   else:
     pm    = 1

   t   = np.tan(np.pi/4.e0-lat  /2.e0) / ( (1.e0-ex*np.sin(lat)  ) / (1.e0+ex*np.sin(lat)   ) )**(ex/2.e0)
   t_c = np.tan(np.pi/4.e0-lat_c/2.e0) / ( (1.e0-ex*np.sin(lat_c)) / (1.e0+ex*np.sin(lat_c) ) )**(ex/2.e0)
   m_c = np.cos(lat_c) / np.sqrt( 1.e0 - ex**2.e0 * (np.sin(lat_c))**2.e0 )
   rho = RT * m_c * t / t_c  #- true scale at lat lat_c

   # outputs :
   x =  pm * rho * np.sin(lon-lon_0)
   y =  -pm * rho * np.cos(lon-lon_0)
   return [x,y]


############################################################################
############################################################################
def xy_to_lonlat(x_in,y_in,RT=6378137.000,ex=0.08181919,lat_true=-71.000,lon_posy=0.e0):
   """
     Converts stereographic (x,y) to (lon,lat) coordinates.

     Usage :

        import stereo
        lon=stereo.xy_to_lonlat(x_in,y_in)[0]
        lat=stereo.xy_to_lonlat(x_in,y_in)[1]

     Optional arguments (default: WGS84):
        RT : Earth radius (m)
        ex : excentricity (0-1)
        lat_true : latitude of true scale
        lon_posy : meridian of positive Y axis

     History: N. Jourdain, IGE-CNRS, April 2019 (from Andy Bliss' matlab)

   """

   #- input checking :
   if ( (RT < 6.e6) | (RT > 7.e6) ):
     sys.exit("~!@#$%^* error : Earth radius is wrong >>>>>> stop !!")
   #-
   if ( (ex < 0.0) | (ex > 1.0) ):
     sys.exit("~!@#$%^* error : Earth excentricity is wrong >>>>>> stop !!")
   #-
   if ( (lat_true < -90.0) | (lat_true > 90.0) ):
     sys.exit("~!@#$%^* error : latitude of true scale is wrong >>>>>> stop !!")
   #-
   if ( (lon_posy < -180.0) | (lon_posy > 360.0) ):
     sys.exit("~!@#$%^* error : meridian of positive Y axis is wrong >>>>>> stop !!")

   #- convert to radians :
   deg2rad = np.pi / 180.e0
   lat_c = deg2rad * lat_true
   lon_0 = deg2rad * lon_posy

   #- if the standard parallel is in S.Hemi., switch signs.
   if ( lat_c < 0.0 ):
     pm    = -1    # plus or minus, north lat. or south
     lat_c = -lat_c
     lon_0 = -lon_0
     x_in  = -x_in
     y_in  = -y_in
   else:
     pm    = 1

   t_c = np.tan(np.pi/4.e0-lat_c/2.e0) / ( (1.e0-ex*np.sin(lat_c)) / (1.e0+ex*np.sin(lat_c)) )**(ex/2.e0)
   m_c = np.cos(lat_c) / np.sqrt( 1.e0 - ex**2.e0 * (np.sin(lat_c))**2.e0 )
   rho = np.sqrt(x_in**2.e0+y_in**2.e0)
   t   = rho * t_c / ( RT * m_c )

   chi = 0.5*np.pi - 2.e0 * np.arctan(t)

   lat = chi + ( (1.e0/2.e0) * ex**2.e0 + (5./24.) * ex**4.e0 + ( 1./ 12.) * ex**6.e0 + (  13./   360.) * ex**8.e0 ) * np.sin(2.e0*chi) \
             + (                          (7./48.) * ex**4.e0 + (29./240.) * ex**6.e0 + ( 811./ 11520.) * ex**8.e0 ) * np.sin(4.e0*chi) \
             + (                                                ( 7./120.) * ex**6.e0 + (  81./  1120.) * ex**8.e0 ) * np.sin(6.e0*chi) \
             + (                                                                        (4279./161280.) * ex**8.e0 ) * np.sin(8.e0*chi)
   lon = lon_0 + np.arctan2(x_in,-y_in)

   lat = pm * lat
   lon = pm * lon
   lon = np.mod(lon+np.pi,2.e0*np.pi)-np.pi # longitude in [-pi,pi]
 
   lon = lon / deg2rad
   lat = lat / deg2rad

   return [lon,lat]
