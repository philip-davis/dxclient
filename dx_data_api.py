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
    
    def _put(self, url, data, files):
        req_url=f'http://{self.socket}/{url}'
        return(requests.put(req_url, data=data, files=files))
    
    def _post(self, url, data, files):
        req_url=f'http://{self.socket}/{url}'
        return(requests.post(req_url, data=data, files=files))

    def _post_json(self, url, json):
        req_url=f'http://{self.socket}/{url}'
        return(requests.post(req_url, json=json))

    def GetNDArray(self, name, version, lb, ub, nspace = None):
        box = _bounds_to_box(lb, ub)
        url = f'/dspaces/obj/{name}/{version}'
        if nspace:
            url = url + f'?namespace={nspace}'
        response = self._post_json(url, box)
        dims = tuple([int(x) for x in response.headers['x-ds-dims'].split(',')])
        tag = int(response.headers['x-ds-tag'])
        dtype = np.sctypeDict[tag]
        arr = np.ndarray(dims, dtype=dtype, buffer=response.content)
        return(arr)

    def PutNDArray(self, name, version, offset, arr, nspace = None):
        box = shape_to_box(arr.shape, offset)
        data = {'box': json.dumps(box)}
        files = {'data': arr.tobytes()}
        url = f'dspaces/obj/{name}/{version}?element_size={arr.itemsize}&element_type={arr.dtype.num}'
        if nspace:
            url = url + f'&namespace={nspace}'
        result = self._put(url, data, files)
    
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
        return(dill.loads(response.content))

if  __name__ == "__main__":
    def test_fn(a, b):
        return(a+b)
    a = np.arange(99).reshape((9, 11))
    b = np.arange(72).reshape((9, 8))
    print(a[2:5,7:9])
    print(b[4:7,1:3])
    print(a[2:5,7:9] + b[4:7,1:3])
    conn = DXDataAPI('20.84.58.28:8000')
    conn.PutNDArray('ex_api2', 1, (2,1), a)
    conn.PutNDArray('ex_api3', 3, (1,0), b)
    result = conn.Exec([
                    ExecArg('ex_api2', 1, (4, 8), (6, 9)),
                    ExecArg('ex_api3', 3, (5, 1), (7, 2))],
                    test_fn)
    print(result)

