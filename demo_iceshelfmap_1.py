import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import stereo
import iceshelfmap as isf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#file_data='/Users/jourdain/XYLAR_ISMIP6_DATA/IMBIE2/rignot_melt_rates_8km.nc'
file_data='/Users/jourdain/XYLAR_ISMIP6_DATA/IMBIE2/bedmap2_8km.nc'

print 'Reading ', file_data
nc1 = xr.open_dataset(file_data)

x=nc1['x'].values[:]
y=nc1['y'].values[:]
surface=nc1['surface'].values[:,:]
#surface=nc1['melt_actual'].values[:,:]

[xnew,ynew,lonnew,latnew,mask,Mnew]=isf.remap(x,y,surface,orientation='portrait') 
#[xnew,ynew,lonnew,latnew,mask,Mnew]=isf.remap(x,y,surface,orientation='dev') 

[x2d,y2d]=np.meshgrid(xnew,ynew)

lat1 = latnew*1.e0; lon1 = lonnew*1.e0
lat2 = latnew*1.e0; lon2 = lonnew*1.e0
lat3 = latnew*1.e0; lon3 = lonnew*1.e0
lat4 = latnew*1.e0; lon4 = lonnew*1.e0
lat1[ mask != 1. ]=np.nan; lon1[ mask != 1. ]=np.nan
lat2[ mask != 2. ]=np.nan; lon2[ mask != 2. ]=np.nan
lat3[ mask != 3. ]=np.nan; lon3[ mask != 3. ]=np.nan
lat4[ mask != 4. ]=np.nan; lon4[ mask != 4. ]=np.nan

#-------------------------------------------------
fig, ax = plt.subplots()

# main fill plot and colorbar
cm = plt.cm.get_cmap('RdYlBu_r')
plt.contourf(x2d,y2d,Mnew,cmap=cm)
cbaxes = inset_axes(ax, width="4%", height="14%", loc=3)
cbar=plt.colorbar(cax=cbaxes, orientation='vertical')
cbar.ax.tick_params(labelsize=5)
plt.sca(ax) # reset current axis to ax

# show mask contours for this projection:
plt.contour(x2d,y2d,mask,[0.5,1.5,2.5,3.5,4.5],colors=('black',),linewidths=(1.5,))

# lon, lat vertices :
levels = [i*1.0 for i in range(-180,180,20)]
plt.contour(x2d,y2d,lon1,levels,colors=('black',),linewidths=(0.5,),linestyles='dotted')
plt.contour(x2d,y2d,lon2,levels,colors=('black',),linewidths=(0.5,),linestyles='dotted')
plt.contour(x2d,y2d,lon3,levels,colors=('black',),linewidths=(0.5,),linestyles='dotted')
plt.contour(x2d,y2d,lon4,levels,colors=('black',),linewidths=(0.5,),linestyles='dotted')
levels = [i*1.0 for i in range(-85,-50,5)]
plt.contour(x2d,y2d,lat1,levels,colors=('black',),linewidths=(0.5,),linestyles='dotted')
plt.contour(x2d,y2d,lat2,levels,colors=('black',),linewidths=(0.5,),linestyles='dotted')
plt.contour(x2d,y2d,lat3,levels,colors=('black',),linewidths=(0.5,),linestyles='dotted')
plt.contour(x2d,y2d,lat4,levels,colors=('black',),linewidths=(0.5,),linestyles='dotted')

# scale :
plt.plot([8.5e5,8.5e5,13.5e5,13.5e5],[2.05e6,2.e6,2.e6,2.05e6],color='black',linewidth=0.8)
plt.text(11.e5,1.95e6,'500 km',fontsize=5,verticalalignment='top',horizontalalignment='center')

#--
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
#asp=((np.max(ynew)-np.min(ynew))/(np.max(xnew)-np.min(xnew)))
ax.set_aspect(1.0)
print '[Ok]'

#--
fig.savefig('test_fig.pdf')
fig.savefig('test_fig.jpg',dpi=300)
plt.show()
