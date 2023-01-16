import os
import os.path as osp
import numpy as np
import scipy.io as sio
import itertools
from copy import deepcopy
import json

def squeeze_list(list_data, axis=None, _now_axis=0):
    """
    REFERENCE: https://aistudio.baidu.com/aistudio/projectdetail/1489965
    
    remove the list dimension with length 1.
     Parameters:
         list_data (list): data in list form
         axis (None, int, list/tuple of int): The axis to be removed, None denotes remove all dimension with length 1.
          Defaults to None.
         _now_axis: Do not change this parameter, this is an inner parameter.
     """
    if axis is None:
        axis = [i for i in range(1000)]
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(list_data, (list, tuple)):
        if len(list_data) == 1 and _now_axis in axis:
            list_data = squeeze_list(list_data[0], axis, _now_axis+1)
        else:
            for i in range(len(list_data)):
                if isinstance(list_data[i], (list, tuple)):
                    try:
                        list_data[i] = squeeze_list(list_data[i], axis, _now_axis+1)
                    except TypeError:
                        pass
                else:
                    pass
    else:
        pass
    return list_data


def MPII_mat2dict(matobj):
    """
    convert MPII .mat format annotations into dict using recursive way
    """
    # if is a number or dict, return directly
    if not isinstance(matobj, np.ndarray) and \
        not isinstance(matobj, sio.matlab.mio5_params.mat_struct):
        objtype = str(type(matobj))
        if 'numpy' in objtype:
            if 'int' in objtype:
                matobj = int(matobj)
            elif 'str' in objtype:
                matobj = str(matobj)
            else:
                assert 'float' in objtype
                matobj = float(matobj)
        return matobj
    
    # if is an array, iterate it and form a new list
    if isinstance(matobj, np.ndarray):
        arr_ls = []
        for eid in range(len(matobj)):
            ret = MPII_mat2dict(matobj[eid])
            arr_ls.append(ret)
        arr_ls = squeeze_list(arr_ls)
        return arr_ls
    
    # if is matobj, just turn to dict with values converted too
    dictionary = {}
    if isinstance(matobj, sio.matlab.mio5_params.mat_struct):
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            cur_val = MPII_mat2dict(elem)
            dictionary[strg] = cur_val
    return dictionary

def jsonify_MPII_anno(mat_path, json_path):
    rel = sio.loadmat(mat_path, struct_as_record=False)['RELEASE'][0, 0]
    mpii_dict = MPII_mat2dict(rel)
    with open(json_path, 'w') as fjson:
        json.dump(mpii_dict, fjson, indent=4)
    print('mat --> json conversion sucess, main keys : ')
    print(mpii_dict.keys())


if __name__ == "__main__":
    mat_path = '/home/jzsherlock/my_lab/storage/pose/public_dataset/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
    json_path = './mpii_json/mpii_anno.json'
    os.makedirs(osp.dirname(json_path), exist_ok=True)
    jsonify_MPII_anno(mat_path=mat_path, json_path=json_path)