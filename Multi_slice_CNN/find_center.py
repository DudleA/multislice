import numpy as np


def find_center(array):
    """
    Find approximate center of segmentation
    """
    x_max, y_max, z_max = array.shape
    x_tot, y_tot = (0, 0)
    x_count, y_count = (0, 0)
    
    for y in range(y_max):
        temp = np.sum(array[:, y, :])
        y_tot += (y * temp)
        y_count += temp
    if y_count == 0:
        raise ValueError("Segmentation is zero (y)")
    else:
        y_mean = y_tot / y_count

    for x in range(x_max):
        temp = np.sum(array[x, :, :])
        x_tot += (x * temp)
        x_count += temp
    if x_count == 0:
        raise ValueError("Segmentation is zero (x)")
    else:
        x_mean = x_tot / x_count

    return int(x_mean), int(y_mean)


def correct_size(array, distance):
    """
    Zero padding
    """
    x_max, y_max, z_max = array.shape
    dx = distance[0]
    dy = distance[1]
    if 2 * dx > x_max:
        diff_x = dx - int(x_max / 2)
        add_x = np.zeros((diff_x, y_max, z_max))
        array = np.append(array, add_x, axis=0)
        array = np.append(add_x, array, axis=0)
        x_max = array.shape[0]

    if 2 * dy > y_max:
        diff_y = dy - int(y_max / 2)
        add_y = np.zeros((x_max, diff_y, z_max))
        array = np.append(array, add_y, axis=1)
        array = np.append(add_y, array, axis=1)
        y_max = array.shape[1]
        
    return array
        

def correct_size3D(array, distance, z_mean):
    """
    Zero padding in 3D
    """
    x_max, y_max, z_max = array.shape
    (dx, dy, dz) = distance
    if 2 * dx > 0.9 * x_max:
        diff_x = dx - int(0.45 * x_max)
        add_x = np.zeros((diff_x, y_max, z_max))
        array = np.append(array, add_x, axis=0)
        array = np.append(add_x, array, axis=0)
        x_max = array.shape[0]

    if 2 * dy > y_max:
        diff_y = dy - int(0.45 * y_max)
        add_y = np.zeros((x_max, diff_y, z_max))
        array = np.append(array, add_y, axis=1)
        array = np.append(add_y, array, axis=1)
        y_max = array.shape[1]
        
    if z_mean < dz:
        diff_z = dz - z_mean
        add_z = np.zeros((x_max, y_max, diff_z))
        array = np.append(add_z, array, axis=2)
        z_mean += diff_z
        
    if z_mean + dz > z_max:
        diff_z = dz + z_mean - z_max
        add_z = np.zeros((x_max, y_max, diff_z))
        array = np.append(array, add_z, axis=2)

    return array, z_mean


def correct_center(center, seg_array, distance, center_seg):
    """
    Correct crop center so that
        - crop is inside image
        - tumor is inside crop
    """
    
    x, y = center
    x_max, y_max, z_max = seg_array.shape
    dx, dy = distance
    x_seg, y_seg = center_seg
    
    
    'Shift crop into the image'
    while (x - dx < 0) & (x + dx < x_max):
        x += 1
    while (x_max - x < dx) & (x - dx > 0):
        x -= 1
    while ((y - dy < 0) & (y + dy < y_max)):
        y += 1
    while ((y_max - y < dy) & (y - dy > 0)):
        y -= 1

    'Shift crop until it contains whole tumor (except outliers)'
    sum_seg = np.sum(seg_array)
    small_dx, small_dy = int(0.95 * dx), int(0.95 * dy)
    while (abs(sum_seg - np.sum(seg_array[x - small_dx:x + small_dx, y - small_dy:y + small_dy, :])) > 0.01 * sum_seg):
        if (x, y) == (x_seg, y_seg):
            raise ValueError("Seg doesn't fit in crop")
        else:
            x += np.sign(x_seg - x) * np.ceil(abs(x_seg - x) / 10).astype(int)
            y += np.sign(y_seg - y) * np.ceil(abs(y_seg - y) / 10).astype(int)
        if (x < 0) | (x  > x_max) | (y < 0) | (y > y_max):
            raise ValueError("Center outside of image")
    if (x - dx < 0) | (x + dx > x_max) | (y - dx < 0) | (y + dy > y_max):
        raise ValueError("Crop outside of image, center: ",x,y, ", max: ", x_max, y_max)
    return x, y
