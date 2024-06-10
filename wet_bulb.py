import numpy as np

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

def WetBulbArrays(t, p, h):
    SHR_CONST_TKFRZ = 273.15
    lambd_a = 3.504     # Inverse of Heat Capacity
    alpha = 17.67 	    # Constant to calculate vapour pressure
    beta = 243.5		# Constant to calculate vapour pressure
    epsilon = 0.6220	# Conversion between pressure/mixing ratio
    es_C = 611.2		# Vapour Pressure at Freezing STD (Pa)
    y0 = 3036		    # constant
    y1 = 1.78		    # constant
    y2 = 0.448		    # constant
    Cf = SHR_CONST_TKFRZ	# Freezing Temp (K)
    p0 = 100000	    # Reference Pressure (Pa)
    constA = 2675 	 # Constant used for extreme cold temperatures (K)
    vkp = 0.2854	 # Heat Capacity

    def QSat_2(T_k, p_t, p0ndplam):
        # Constants used to calculate es(T)
        # Clausius-Clapeyron
        tcfbdiff = T_k - Cf + beta
        es = es_C * np.exp(alpha*(T_k - Cf)/(tcfbdiff))
        dlnes_dT = alpha * beta/((tcfbdiff)*(tcfbdiff))
        pminuse = p_t - es
        de_dT = es * dlnes_dT

        # Constants used to calculate rs(T)
        rs = epsilon * es/(p0ndplam - es + np.spacing(1)) #eps

        # avoid bad numbers
        if rs > 1 or rs < 0:
            rs = np.nan

        return es,rs,dlnes_dT

    def DJ(T_k, p_t, p0ndplam):
        # Constants used to calculate es(T)
        # Clausius-Clapeyron
        tcfbdiff = T_k - Cf + beta
        es = es_C * np.exp(alpha*(T_k - Cf)/(tcfbdiff))
        dlnes_dT = alpha * beta/((tcfbdiff)*(tcfbdiff))
        pminuse = p_t - es
        de_dT = es * dlnes_dT

        # Constants used to calculate rs(T)
        rs = epsilon * es/(p0ndplam - es + np.spacing(1)) #eps)
        prersdt = epsilon * p_t/((pminuse)*(pminuse))
        rsdT = prersdt * de_dT

        # Constants used to calculate g(T)
        rsy2rs2 = rs + y2*rs*rs
        oty2rs = 1 + 2.0*y2*rs
        y0tky1 = y0/T_k - y1
        goftk = y0tky1 * (rs + y2 * rs * rs)
        gdT = - y0 * (rsy2rs2)/(T_k*T_k) + (y0tky1)*(oty2rs)*rsdT

        # Calculations used to calculate f(T,ndimpress)
        foftk = ((Cf/T_k)**lambd_a)*(1 - es/p0ndplam)**(vkp*lambd_a)*         np.exp(-lambd_a*goftk)
        fdT = -lambd_a*(1.0/T_k + vkp*de_dT/pminuse + gdT) * foftk

        return foftk,fdT

    def WetBulb(TemperatureK,Pressure,Humidity):
        HumidityMode = 0

        pnd = (Pressure/p0)**(vkp)
        p0ndplam = p0*pnd**lambd_a

        C = SHR_CONST_TKFRZ;		# Freezing Temperature
        T1 = TemperatureK;		# Use holder for T

        if T1 > 10e6 or Humidity > 10e6:
            return np.nan

        es, rs, _ = QSat_2(TemperatureK, Pressure, p0ndplam) # first two returned values

        if HumidityMode==0:
            qin = Humidity                   # specific humidity
            relhum = 100.0 * qin/rs          # relative humidity (%)
            vape = es * relhum * 0.01   # vapor pressure (Pa)
        elif HumidityMode==1:
            relhum = Humidity                # relative humidity (%)
            qin = rs * relhum * 0.01         # specific humidity
            vape = es * relhum * 0.01   # vapor pressure (Pa)

        mixr = qin * 1000          # change specific humidity to mixing ratio (g/kg)

        # Calculate Equivalent Pot. Temp (Pressure, T, mixing ratio (g/kg), pott, epott)
        # Calculate Parameters for Wet Bulb Temp (epott, Pressure)
        D = 1.0/(0.1859*Pressure/p0 + 0.6512)
        k1 = -38.5*pnd*pnd + 137.81*pnd - 53.737
        k2 = -4.392*pnd*pnd + 56.831*pnd - 0.384

        # Calculate lifting condensation level
        tl = (1.0/((1.0/((T1 - 55))) - (np.log(relhum/100.0)/2840.0))) + 55.0

        # Theta_DL: Bolton 1980 Eqn 24.
        theta_dl = T1*((p0/(Pressure-vape))**vkp) * ((T1/tl)**(mixr*0.00028))
        # EPT: Bolton 1980 Eqn 39.
        epott = theta_dl * np.exp(((3.036/tl)-0.00178)*mixr*(1 + 0.000448*mixr))
        Teq = epott*pnd	# Equivalent Temperature at pressure
        X = (C/Teq)**3.504

        # Calculates the regime requirements of wet bulb equations.
        invalid = Teq > 600 or Teq < 200
        hot = Teq > 355.15
        cold = X>=1 and X<=D
        if invalid:
            return np.nan

        # Calculate Wet Bulb Temperature, initial guess
        # Extremely cold regimes: if X.gt.D, then need to calculate dlnesTeqdTeq

        es_teq, rs_teq, dlnes_dTeq = QSat_2(Teq, Pressure, p0ndplam)
        if X<=D:
            wb_temp = C + (k1 - 1.21 * cold - 1.45 * hot - (k2 - 1.21 * cold) * X + (0.58 / X) * hot)
        else:
            wb_temp = Teq - ((constA*rs_teq)/(1 + (constA*rs_teq*dlnes_dTeq)))

        # Newton-Raphson Method
        maxiter = 2
        iter = 0
        delta = 1e6

        while delta>0.01 and iter<maxiter:
            foftk_wb_temp, fdwb_temp = DJ(wb_temp, Pressure, p0ndplam)
            delta = (foftk_wb_temp - X)/fdwb_temp  #float((foftk_wb_temp - X)/fdwb_temp)
            delta = np.minimum(10,delta)
            delta = np.maximum(-10,delta) #max(-10,delta)
            wb_temp = wb_temp - delta
            Twb = wb_temp
            iter = iter+1

        return Twb-C

    import numpy as np
    vWetBulb = np.vectorize(WetBulb)
    result = np.empty_like(t)
    if len(result.shape) > 2:
        for d in range(result.shape[0]):
            res_slice = vWetBulb(t[d], p, h[d])
            print(res_slice.shape)
            result[d] = vWetBulb(t[d], p, h[d])
    else:
        result = vWetBulb(t, p, h)
    return(result)
