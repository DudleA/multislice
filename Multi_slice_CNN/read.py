import numpy as np
import SimpleITK as sitk
import pandas as pd
import keras
import random
import os, glob

from find_center import (find_center, correct_center, correct_size,
    correct_size3D)


def xReadFunction(ID, params, im_mask="both", data=None):
    """
    Read and crop MR image and segmentation mask
    """
    
    'Read path of image file for the patient from a dataframe'
    source = params["source"]
    if data == "dataset_name"
        df = pd.read_csv(os.path.join(source, 'dataframe.csv'), index_col=0, header=0)
    path = df.loc[ID[0], 'Patient path']
    else: 
        raise ValueError("Dataset name missing")
    
    'Read image and segmentation according to spacing and norm parameters'
    if params["spacing"] == "same":
        if params["norm"] == "slice":
            img = sitk.ReadImage(os.path.join(path, 'norm4_spacing_img.nii.gz'))
        elif params["norm"] == "scan":
            img = sitk.ReadImage(os.path.join(path, 'norm2_spacing_img.nii.gz'))
        elif params["norm"] == "clip":
            img = sitk.ReadImage(os.path.join(path, 'clip_spacing_img.nii.gz'))
        else:
            img = sitk.ReadImage(os.path.join(path, 'same_spacing_img.nii.gz'))

        seg = sitk.ReadImage(os.path.join(path, 'same_spacing_seg.nii.gz'))
    else:
        if params["norm"] == "slice":
            img = sitk.ReadImage(os.path.join(path, 'norm4_image.nii.gz'))
        elif params["norm"] == "scan":
            img = sitk.ReadImage(os.path.join(path, 'norm2_image.nii.gz'))
        elif params["norm"] == "clip":
            img = sitk.ReadImage(os.path.join(path, 'clip_image.nii.gz'))
        else:
            img = sitk.ReadImage(os.path.join(path, 'image.nii.gz'))

        seg = glob.glob(os.path.join(path, 'seg*.nii.gz'))
        seg = sitk.ReadImage(seg[0])

        img_array = sitk.GetArrayFromImage(img)
        seg_array = sitk.GetArrayFromImage(seg)
        
        z_mean = int(float(ID[1]))
        

    (dx, dy, dz) = params["distance"]
    img_array = np.transpose(img_array, (2, 1, 0))
    seg_array = np.transpose(seg_array, (2, 1, 0))
    
    
    if ID.shape[0] < 3:
        'zero padding if necessary'
        img_array, temp = correct_size3D(img_array, (dx,dy,dz), z_mean)
        seg_array, z_mean = correct_size3D(seg_array, (dx,dy,dz), z_mean)
    
        if temp!=z_mean:
            raise ValueError("Slice number of image and segmentation not equal")
    
        'if minimum and maximum slices are not provided, crop to a fixed number of slices'
        img_array = img_array[:, :, z_mean-dz:z_mean+dz]
        seg_array = seg_array[:, :, z_mean-dz:z_mean+dz]
    
    elif ID.shape[0] == 3:
        z_min = int(float(ID[1]))
        z_max = int(float(ID[2]))
        z_mean = int(np.rint((z_min + z_max)/2))
        dz = int(np.ceil((z_max-z_min)/2))
        
        'zero padding if necessary'
        img_array, temp = correct_size3D(img_array, (dx,dy,dz), z_mean)
        seg_array, z_mean = correct_size3D(seg_array, (dx,dy,dz), z_mean)
    
        if temp!=z_mean:
            raise ValueError("Slice number of image and segmentation not equal")
        
        'crop image between minimum and maximum slices'
        img_array = img_array[:, :, z_min:z_max+1]
        seg_array = seg_array[:, :, z_min:z_max+1]
    
    else:
        raise ValueError("Slice not defined, ID: ", ID)
        
    x_seg, y_seg = find_center(seg_array)

    'Segmentation center'
    if params["centered"] == "seg":
        x, y = x_seg, y_seg
    
    'Random crop center'
    else:
        x_max, y_max, z_max = img_array.shape
        x = int(x_max / 2) + random.randint(-int(x_max / 20), int(x_max / 20))
        y = int(y_max / 2) + random.randint(-int(y_max / 20), int(y_max / 20))
            
    'Correct crop center so that it contains the whole tumor'
    x, y = correct_center((x, y), seg_array, (dx, dy), (x_seg, y_seg))
    
    'Cropping'
    img_array = img_array[x - dx:x + dx, y - dy:y + dy, :]
    seg_array = seg_array[x - dx:x + dx, y - dy:y + dy, :]

    if len(params["distance"]) == 3:
        img_array = np.expand_dims(img_array, axis = 3)
        seg_array = np.expand_dims(seg_array, axis = 3)
        return img_array, seg_array, z_mean
   
    else:
        return img_array, seg_array



def yReadFunction(ID, params):
    """
    Read image labels
    """

    source = params["source"]
    data = params["data"]
    label = params["label"]
    spacing = params["spacing"]
    
    if data == 'dataset_name':
        df = pd.read_csv(os.path.join(source, 'dataframe.csv'), index_col=0, header=0)
    else: 
        raise ValueError("Dataset name missing")
        
    if len(ID.shape) == 2:
        y = np.empty(0, dtype=int)
        for j in range(ID.shape[0]):
            temp_label = df.loc[ID[j, 0], 'Label']
            y = np.append(y, temp_label)
    else:
        y = df.loc[ID[0], 'Label']
    return y



def ReadPartition(source, params, col=None):
    """
    Read index col of existing split
    """
    label = params["label"]
    data = params["data"]
    spacing = params["spacing"]
    distance = params["distance"]

    if spacing == "same":
        df = pd.read_csv(os.path.join(source, 'partitions_multi_same.csv'), index_col=0, header=0)
    elif spacing == "original":
        df = pd.read_csv(os.path.join(source, 'partitions__multi_original.csv'), index_col=0, header=0)
    else:
        raise NameError("Partition missing")

    df.columns = df.columns.astype(int)
    
    if col is None:
        y = df.shape[1]
        col = random.randint(0, y / 2 - 1)
    partition = {}
    a = df.loc[:, 2 * col]
    partition["train"] = a.dropna()
    a = df.loc[:, 2 * col + 1]
    partition["validation"] = a.dropna()
    return partition
        

