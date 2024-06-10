from dx_data_api import DXDataAPI, ExecArg
from datetime import date
from dateutil import parser
from bitstring import pack
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import netCDF4 as nc

def _discretize_geo(lat, lon):
    assert lat >= -60 and lat <= 90
    assert lon >= -180 and lon <= 180 
    if lat >= 89.875:
        lat = lat - 0.125
    if lon >= 179.875:
        lon = lon - 0.125
    return (int((lat+60.0)/0.25), int((lon+180.0)/0.25))

def _build_var_name(variable, model = None, scenario = None):
    var_name = f'v:{variable}'
    if model:
        var_name = var_name + f',m:{model}'
    if scenario:
        var_name = var_name + f',s:{scenario}'
    return var_name

def _get_version(start_date, end_date):
    s = parser.parse(start_date).date()
    e = parser.parse(end_date).date()
    assert(e >= s)
    base_date = date(1950,1,1)
    start = (s - base_date).days
    span = (e - s).days
    version = pack('uint:16, uint:16', start, span).uint
    return(version)

class DXInterface:
    def __init__(self, socket):
       self.api = DXDataAPI(socket)


    def _query_pc(self, variable, start_date, end_date, model = None, scenario = None, geo_lb = (-60.0,-180.0), geo_ub = (90.0,180.0)):
        lb = _discretize_geo(*geo_lb)
        ub = _discretize_geo(*geo_ub)
        assert lb[0] <= ub[0] and lb[1] <= ub[1]
        var_name = _build_var_name(variable, model, scenario)
        version = _get_version(start_date, end_date)
        data = self.api.GetNDArray(var_name, version, lb, ub, nspace = 'cmip6-planetary')
        return(data)

    def query(self, source, **kwargs):
        if source == 'planetary-gddp':
            return(self._query_pc(**kwargs))

    def write(self, variable, model, data, **kwargs):
        self.api.PutNDArray(f'v:{variable},m:{model}', 0, (0,0), data)

    def _build_arg_pc(variable, start_date, end_date, model = None, scenario = None, geo_lb = (-60.0,-180.0), geo_ub = (90.0,180.0)):
        lb = _discretize_geo(*geo_lb)
        ub = _discretize_geo(*geo_ub)
        assert lb[0] <= ub[0] and lb[1] <= ub[1]
        var_name = _build_var_name(variable, model, scenario)
        version = _get_version(start_date, end_date)
        return(ExecArg(var_name, version, lb, ub, 'cmip6-planetary'))

    def _build_arg_local(variable, model = None, geo_lb = (-60.0,-180.0), geo_ub = (90.0,180.0)):
        if model == 'mymodel':
            lb = _discretize_geo(*geo_lb)
            ub = _discretize_geo(*geo_ub)
            assert lb[0] <= ub[0] and lb[1] <= ub[1]
        else:
            raise ValueError
        return(ExecArg(f'v:{variable},m:{model}', 0, lb, ub))

    def build_arg(source, **kwargs):
        if source == 'planetary-gddp':
            return(DXInterface._build_arg_pc(**kwargs))
        elif source == 'local':
            return(DXInterface._build_arg_local(**kwargs))

    def exec(self, fn, args):
        return(self.api.Exec(args, fn))

if __name__ == "__main__":
    client = DXInterface('20.84.58.28:8000')
    data = client.query(source = 'planetary-gddp',
		variable = 'tas',
		model = 'ACCESS-ESM1-5',
		start_date = '1982-11-28',
		end_date = '1982-11-29',
		geo_lb = (38.9,-77.0),
		geo_ub = (40.7,-74.0))
    print(data.shape)
    print(data)
    def pressurefromelev(elev):
        #Necessary constants
        Tbase=288;      #temperature at base of atmosphere -- K
        L=-6.5;         #lapse rate -- K/km
        G=-9.81;        #gravity -- m/s^2
        M=28.96;        #density of air -- kg/mol
        R=8.314;        #universal gas constant -- J/(kg*mol)

        #Note that mean global sea-level pressure over land is approx 1010 hPa
        pressure=100*np.round(1010*((Tbase+L*10**-3*elev)/Tbase)**((G*M)/(R*L)),2);
        return pressure

    f=nc.Dataset('elev_721x1440.nc'); #geopotential
    z=f['z'][:,:,:]/9.81;z_orig=z[0,:,:]; #converted to meters, centered on 180W

    lat_cur=np.linspace(-90,90,721); #because elev data file is 721x1440
    lon_cur=np.linspace(-180,180,1440);
    models=['ACCESS-ESM1-5'];
    modellatsz=[600];
    modellonsz=[1440];


    for m in range(len(models)):
        lat_des=np.linspace(-90,90,modellatsz[m])
        lon_des=np.linspace(-180,180,modellonsz[m]);
        interp = RegularGridInterpolator((lat_cur, lon_cur), z_orig,bounds_error=False,fill_value=None);
        X,Y = np.meshgrid(lat_des,lon_des,indexing='ij')
        z_interp=interp((X,Y))
        psfc_chunk=pressurefromelev(z_interp[:,:]);
    pressure = psfc_chunk
    
    client.write(variable = 'pressure',
	    model = 'mymodel',
	    geo_resolution = (0.25,0.25),
        projection = 'wgs84',
        geo_offset = (-60.0, -180.0),
	    data = pressure)
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
    
    def HeatIndex(t, p, h):
        def HeatIndexKernel(t1, p1, h1):
            return(t1, p1, h1)
        import numpy
        HeatIndexV = numpy.vectorize(HeatIndexKernel)
        return(HeatIndexV(t, p, h))
    result = client.exec(HeatIndex, [argT, argP, argH])
    print(result)

    

        
	    


