"""
Generate stratified splits
Two conditions: 
- Conserving ratios between phenotypes
- Same training/validation ratio for patients and for slices
"""

import random
import copy
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os, glob
from read import yReadFunction

def GetFileLists_stratified(params, shuffle=True):
    data = params["data"]
    source = params["source"]
    if data == 'dataset':
        df=pd.read_csv(os.path.join(source,
            'dataframe.csv'), 
            index_col=0, header=0)
    else:
        raise NameError("Dataset name unknown")
    
    List_IDs=copy.deepcopy(df.index)
    List_IDs=np.asarray(List_IDs)
    
    len_tot = List_IDs.shape[0]
    len_train = np.rint(len_tot*0.8)
    
    label = df.loc[:,"NumLabel"]
    hcc = List_IDs[label == 4]
    hca = List_IDs[label == 0]
    fnh = List_IDs[label == 1]
    cca = List_IDs[label == 3]
    
    print(hcc.shape[0], hca.shape[0], fnh.shape[0], cca.shape[0])
    len_hcc = int(np.rint(0.8*hcc.shape[0]))
    len_hca = int(np.rint(0.8*hca.shape[0]))
    len_fnh = int(np.rint(0.8*fnh.shape[0]))
    len_cca = int(np.rint(0.8*cca.shape[0]))
    print(len_hcc, len_hca, len_fnh, len_cca)
    
    check_slice = False
    t=0.0
    
    while check_slice is False:
        """Shuffle each phenotype separately"""
        random.shuffle(hcc)
        random.shuffle(hca)
        random.shuffle(fnh)
        random.shuffle(cca)
        
        partition = {}
        partition_multi = {}
        partition_3d = {}
        partition_2d = {}
        
        """Combine phenotypes to get new splits"""
        partition["train"] = np.concatenate([hcc[:len_hcc], hca[:len_hca], 
            fnh[:len_fnh], cca[:len_cca]])
        partition["validation"] = np.concatenate([hcc[len_hcc:], hca[len_hca:], 
            fnh[len_fnh:], cca[len_cca:]])
            
        """Get splits for multi-slice network"""
        partition_multi["train"], len_part_train = slicing(partition["train"], 
                df, shuffle, params)  
        partition_multi["validation"], len_part_val = slicing(partition["validation"], 
            df, shuffle, params)
        
        print(abs(len_part_train/(len_part_train + len_part_val)))
        print(len_train/len_tot)
        
        """Make sure ratio train/validation sets is still similar 
            after slicing,  otherwise generate new split"""
        if (abs(len_part_train/(len_part_train + len_part_val) - len_train/len_tot)> 0.02) :
            check_slice = False
        else: 
            check_slice = True
    
    """When condition fullfilled, get same splits for 2D network"""
    params["distance"] = (180, 180)
    partition_2d["train"], len_part_train = slicing(partition["train"], 
                df, shuffle, params)
    partition_2d["validation"], len_part_val = slicing(partition["validation"], 
            df, shuffle, params)
    
    """Get same splits for 3D network"""
    params["distance"] = (180, 180, 5)
    partition_3d["train"], len_part_train = slicing(partition["train"], 
                df, shuffle, params)  
    partition_3d["validation"], len_part_val = slicing(partition["validation"], 
            df, shuffle, params)
            
    """Back to original parameters for next function call"""        
    params["distance"] = (180, 180, None)
            
    return partition_2d, partition_multi, partition_3d

"""
Collect slices which fit inclusion criteria
Shape of output array depends on params["distance"] shape
"""
def slicing(List_IDs, df,shuffle, params):
    spacing = params["spacing"]
    x = []
    count = 0
    for ID in List_IDs:
        #print(ID)
        path = df.loc[ID,'Patient path']
        #print(path[:-12])
        if spacing == "same":
            seg =  sitk.ReadImage(os.path.join(path, 'same_spacing_seg.nii.gz'))
        elif spacing == "original":
            seg = glob.glob(os.path.join(path, 'seg*.nii.gz'))
            seg = sitk.ReadImage(seg[0])
        seg_array = sitk.GetArrayFromImage(seg)
        n_slices = seg.GetSize()[2]
        max_area = np.max(np.sum(seg_array[:,:,:], axis = (1,2)))
        
        """For 2D model"""
        if len(params["distance"]) == 2:
            for n in range(n_slices):
                if np.sum(seg_array[n,:,:]) > 0.2 * max_area:
                    count += 1
                    x.append(str(ID) +', '+ str(n))
        elif len(params["distance"]) == 3:
            dz = params["distance"][2]
            slices = np.array(ID)
            """For multi-slice model"""
            if dz is None:
                for n in range(n_slices):
                    if (np.sum(seg_array[n,:,:]) > 0.2 * max_area):
                        count += 1
                        slices = np.append(slices, n)
            """For 3D model"""
            else:
                if n_slices <= 2 * dz:
                    slices = np.append(slices, np.rint(n_slices/2))
                    count += 1
                else:
                    for n in range(dz, n_slices-dz):
                        if ((np.sum(seg_array[n,:,:]) > 0.2 * max_area) & 
                            (np.sum(seg_array[n-1, : ,:]) > 0.2 * max_area) & 
                            (np.sum(seg_array[n+1, :, :]) > 0.2 * max_area)):
                                slices = np.append(slices, n)
                                count += 1
                #while (len(slices.shape) < 1) & (dz - i > 1):
                #    n = dz-i
                if (len(slices.shape) < 1):
                    if np.sum(seg_array[:dz,:,:]) > 0:
                        slices = np.append(slices, dz)
                    elif np.sum(seg_array[n_slices-dz:,:,:]) > 0:
                        slices = np.append(slices, n_slices-dz)
                    else:
                        slices = np.append(slices, np.rint(n_slices/2))
                    count += 1
            x.append([slices])
    if shuffle:
        random.shuffle(x)
    return x, count
