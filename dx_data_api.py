import json
from dataclasses import dataclass
import numpy as np
import requests
import dill

@dataclass
class ExecArg:
    name: str
    version: int
    lb: tuple[int, ...]
    ub: tuple[int, ...]
    namespace:str = None

@dataclass
class DSRegHandle:
    namespace: str
    parameters: dict

def _bounds_to_box(lb, ub):
    if len(lb) != len(ub):
        raise TypeError('lb and ub must have same dimensionality')
    box = {}
    box['bounds'] = [{'start': l, 'span': (u-l)+1} for l,u in zip(lb,ub)]
    return(box)

def shape_to_box(shape, offset):
    if len(shape) != len(offset):
        raise TypeError('shape and offset must have same dimensionality')
    box = {}
    box['bounds'] = []
    for s, o in zip(shape, offset):
        box['bounds'].append({'start': o, 'span':s})
    return(box)

class DXDataAPI:
    def __init__(self, socket):
        self.socket = socket
    
    def _req_url(self, url):
        return(f'http://{self.socket}/{url}')

    def _put(self, url, data, files):
        return(requests.put(self._req_url(url), data=data, files=files))
    
    def _post(self, url, data, files = None):
        return(requests.post(self._req_url(url), data=data, files=files))

    def _post_json(self, url, json):
        return(requests.post(self._req_url(url), json=json))

    def _get(self, url):
        return(requests.get(self._req_url(url)))

    def _get_url_content(self, url):
        response = self._get(url)
        if response.status_code == 404:
            return(None)
        if not response.ok:
            raise RuntimeError(f'request to server failed with {response.status_code}.')
        return(json.loads(response.content))

    def GetNDArray(self, name, version, lb, ub, nspace = None):
        box = _bounds_to_box(lb, ub)
        url = f'/dspaces/obj/{name}/{version}'
        if nspace:
            url = url + f'?namespace={nspace}'
        response = self._post_json(url, box)
        if response.status_code == 404:
            return None
        if not response.ok:
            raise RuntimeError(f'request to server failed with {response.status_code}.')
        dims = tuple([int(x) for x in response.headers['x-ds-dims'].split(',')])
        tag = int(response.headers['x-ds-tag'])
        dtype = np.sctypeDict[tag]
        arr = np.ndarray(dims, dtype=dtype, buffer=response.content)
        return(arr)

    def PutNDArray(self, arr, name, version, offset, nspace = None):
        box = shape_to_box(arr.shape, offset)
        data = {'box': json.dumps(box)}
        files = {'data': arr.tobytes()}
        url = f'dspaces/obj/{name}/{version}?element_size={arr.itemsize}&element_type={arr.dtype.num}'
        if nspace:
            url = url + f'&namespace={nspace}'
        response = self._put(url, data, files)
        if not response.ok:
            raise RuntimeError(f'request to server failed with {response.status_code}.')
    
    def Exec(self, args, fn):
        objs = []
        for obj in args:
            box = _bounds_to_box(obj.lb, obj.ub)
            objs.append({
                'name': obj.name,
                'version': obj.version,
                'bounds': box['bounds']})
            if obj.namespace:
                objs[-1]['namespace'] = obj.namespace
        data = {'requests': json.dumps({'requests': objs})}
        files = {'fn': dill.dumps(fn)}
        url = f'dspaces/exec/'
        response = self._post(url, data, files)
        if response.status_code == 404:
            return(None)
        if not response.ok:
            raise RuntimeError(f'request to server failed with {response.status_code}.')
        return(dill.loads(response.content))

    def GetVars(self):
        return(self._get_url_content('dspaces/var/'))

    def GetVarObjs(self, name):
        return(self._get_url_content(f'dspaces/var/{name}/'))

    def Register(self, type, name, data):
        url = f'dspaces/register/{type}/{name}'
        response = self._post(url, json.dumps(data))
        if not response.ok:
            content = json.loads(response.content)
            err_msg = content['detail']
            raise RuntimeError(f'request to server failed with {response.status_code}: {err_msg}.')
        handle_dict = json.loads(response.content)
        return(DSRegHandle(**handle_dict))  

if  __name__ == "__main__":
    def test_fn(a, b):
        return(a+b)
    a = np.arange(99).reshape((9, 11))
    b = np.arange(72).reshape((9, 8))
    print(a[2:5,7:9])
    print(b[4:7,1:3])
    print(a[2:5,7:9] + b[4:7,1:3])
    conn = DXDataAPI('127.0.0.1:8002')
    conn.PutNDArray(a, 'ex_api2', 1, (2,1))
    conn.PutNDArray(b, 'ex_api3', 3, (1,0))
    result = conn.Exec([
                    ExecArg('ex_api2', 1, (4, 8), (6, 9)),
                    ExecArg('ex_api3', 3, (5, 1), (7, 2))],
                    test_fn)
    print(result)
    result = conn.GetVars()
    print(result)
    result = conn.GetVarObjs('ex_api3')
    print(result)
    data = {'bucket':'noaa-goes17', 'path':'ABI-L1b-RadM/2020/215/15/OR_ABI-L1b-RadM1-M6C04_G17_s20202151508255_e20202151508312_c20202151508354.nc'}
    result = conn.Register('s3nc', 'foo', data)
    print(result)
