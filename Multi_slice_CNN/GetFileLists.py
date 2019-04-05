import random
import copy
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os, glob
from read import yReadFunction

def GetFileLists(params, shuffle=True):
    """
    Generate a split with
        - similar class balance in validation and training sets
        - similar average number of slice per patient
    """
    data = params["data"]
    source = params["source"]
    if data == 'dataset_name':
        df=pd.read_csv(os.path.join(source,'dataframe.csv'), 
            index_col=0, header=0)
  
    List_IDs=copy.deepcopy(df.index)
    List_IDs=np.asarray(List_IDs)
    check_slice = False
    len_tot = List_IDs.shape[0]
    len_train = int(len_tot*0.8)
    t=0.0
    
    'Generate splits and check for balance until all conditions are met'
    while check_slice is False & (t<20):
        partition={}
        temp = {}
        random.shuffle(List_IDs)
        partition["train"] = List_IDs[:len_train]
        partition["validation"] = List_IDs[len_train:]
        partition["train"] = slicing(partition["train"], 
            df, shuffle, params)
        partition["validation"]= slicing(partition["validation"], 
            df, shuffle, params)
        
        'Check if average number of slices per patient balanced'
        if (abs(len(partition["train"])/(len(partition["train"])+
            len(partition["validation"])) - len_train/len_tot)> 0.05) :
            check_slice = False
            continue
        'Check if classes are balanced'
        check_slice = check_class_balance(partition, params)
        t+=1.0
        
    if t>=20:
        raise RuntimeError("No partition found")
    else: 
        return partition
    
    
def check_class_balance(partition, params, balance = (0.1,0.1)) : 
    label = params["label"]
    partition["train"] = np.asarray(partition["train"])
    partition["validation"] = np.asarray(partition["validation"])
    train_label = yReadFunction(partition["train"],params)
    val_label = yReadFunction(partition["validation"],params)

    train = np.sum(train_label)/len(train_label)
    val = np.sum(val_label)/len(val_label)
    tot = (np.sum(val_label)+np.sum(train_label))/(len(val_label)+
        len(train_label))
    
    if (abs((tot-train)/tot) > balance[0]) | (abs((tot-val)/tot) > balance[1]):
        return False
    else : return True
    

def slicing(List_IDs, df ,shuffle, params):
    """
    Returns list of slices satisfying inclusion criterion
    slice tumor area > 0.2 * max tumor area
    """
    spacing = params["spacing"]
    x = []
    for ID in List_IDs:
        path = df.loc[ID,'Patient path']
        if spacing == "same":
            seg =  sitk.ReadImage(os.path.join
                (path, 'same_spacing_seg.nii.gz'))
        elif spacing == "original":
            seg = glob.glob(os.path.join(path, 'seg*.nii.gz'))
            seg = sitk.ReadImage(seg[0])
        seg_array = sitk.GetArrayFromImage(seg)
        
        n_slices = seg.GetSize()[2]
        slices = np.array(ID)
        max_area = np.max(np.sum(seg_array[:,:,:], axis = (1,2)))
        for n in range(n_slices):
            if np.sum(seg_array[n,:,:]) > 0.2 * max_area:
                slices = np.append(slices, n)
        x.append(slices)
        
    if shuffle:
        random.shuffle(x)
    return x
        
