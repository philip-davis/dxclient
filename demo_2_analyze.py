from dx_interface import DXInterface
from wet_bulb import pressurefromelev
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator
import numpy as np

# Generate elevation-dependent atmospheric pressure
f=nc.Dataset('elev_721x1440.nc'); #geopotential
z=f['z'][:,:,:]/9.81;z_orig=z[0,:,:]; #converted to meters, centered on 180W
lat_cur=np.linspace(-90,90,721); #because elev data file is 721x1440
lon_cur=np.linspace(-180,180,1440);
lat_des=np.linspace(-90,90,600)
lon_des=np.linspace(-180,180,1440);
interp = RegularGridInterpolator((lat_cur, lon_cur), z_orig,bounds_error=False,fill_value=None);
X,Y = np.meshgrid(lat_des,lon_des,indexing='ij')
z_interp=interp((X,Y))
psfc_chunk=pressurefromelev(z_interp[:,:]);
pressure = psfc_chunk


client = DXInterface('20.84.58.28:8000')

# Write the custom pressure variable
client.write(variable = 'pressure'
	    model = 'mymodel',
	    geo_resolution = (0.25,0.25),
        projection = 'wgs84',
        geo_offset = (-60.0, -180.0),
	    data = pressure)


#Setup argumnets for remote server execution
argT = DXInterface.build_arg(source = 'planetary-gddp',
		variable = 'tas',
		model = 'ACCESS-ESM1-5',
        start_date = '1982-11-28',
        end_date = '1982-11-29',
        geo_lb = (38.9,-77.0),
        geo_ub = (40.7,-74.0))
argP = DXInterface.build_arg(source = 'local',
        variable = 'pressure',
        model = 'mymodel',
        geo_lb = (38.9,-77.0),
        geo_ub = (40.7,-74.0))
argH = DXInterface.build_arg(source = 'planetary-gddp',
        variable = 'huss',
        model = 'ACCESS-ESM1-5',
        start_date = '1982-11-28',
        end_date = '1982-11-29',
        geo_lb = (38.9,-77.0),
        geo_ub = (40.7,-74.0))

#define server-side subroutine
def HeatIndex(t, p, h):
        def foo(tf, v):
            return tf * v
        def HeatIndexKernel(t1, p1, h1):
            return(foo(t1*p1, h1))
        import numpy
        HeatIndexV = numpy.vectorize(HeatIndexKernel)
        return(HeatIndexV(t, p, h))

#Run the remote exeuction operation
result = client.exec(HeatIndex, [argT, argP, argH])
print(result)

