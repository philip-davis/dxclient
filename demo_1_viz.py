from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from dx_interface import DXInterface

# Establish client connection
client = DXInterface('20.84.58.28:8000')

# Query Temperature data - latitude and longitude coordinates are provided that align with the grid
data, lat, lon = client.query(source = 'planetary-gddp',
		variable = 'tas',
		model = 'ACCESS-ESM1-5',
		start_date = '1982-11-28',
		end_date = '1982-11-29',
		geo_lb = (38.9,-77.0),
		geo_ub = (40.7,-74.0))

# Plot a contour map
fig = plt.figure(num=None, figsize=(7, 7) ) 
m = Basemap(projection='cyl', llcrnrlon=-77, llcrnrlat=38.9, urcrnrlon=-74, urcrnrlat=40.7, resolution='i')
x, y = m(*np.meshgrid(lon,lat))
cs = m.contourf(x, y, data[0], levels = 100, cmap=plt.cm.jet)
m.drawcoastlines()
m.drawmapboundary()
m.drawcountries(linewidth=1, linestyle='solid', color='k' ) 
m.drawmeridians(range(33, 48, 2), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
m.drawparallels(range(3, 15, 2), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])
plt.ylabel("Latitude", fontsize=15, labelpad=35)
plt.xlabel("Longitude", fontsize=15, labelpad=20)
cbar = m.colorbar(cs, location='right', pad="3%")
cbar.set_label('Temperature (K)', fontsize=13)
plt.title('near-surface air temperature', fontsize=15)
plt.show()
