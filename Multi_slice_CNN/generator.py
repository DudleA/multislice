# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:25:26 2018


VERSION BUILT TO HANDLE IMAGES WITH A CHANNEL LAST AXIS

@author: fcalvet
under GPLv3

Adapted in March 2019 by A. Dudle for the multi-slice CNN

"""

import numpy as np  # to manipulate the arrays
import tensorflow.keras  # to use the Sequence class
import random
from matplotlib import pyplot as plt  # to create figures examples
import os  # to save figures
import elasticdeform

from multi_channel_image_augmentation import (random_transform, 
    deform_grid, deform_pixel)
from read import yReadFunction


class DataGenerator(tensorflow.keras.utils.Sequence):
    """
    Generates data for Keras
    Based on keras.utils.Sequence for efficient and safe multiprocessing
    idea from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    Needs on initialization:
        list_IDs: a list of ID which will be supplied to the ReadFunction to obtain
        params: a dictionnary of parameters, explanation supplied in the README
        batch_size: number of IDs used to generate a batch
        shuffle: if set to True, the list will be shuffled every time an epoch ends
        plotgenerator: number of times a part of a batch will be saved to disk as examples
    """

    def __init__(self, list_IDs, params, batch_size=1, shuffle=True,
                 balance=False, plotgenerator=0):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs_original = list_IDs
        self.shuffle = shuffle
        self.plotgenerator = plotgenerator
        self.params = params
        self.plotedgenerator = 0  # counts the number of images saved
        self.list_IDs_temp = []
        self.balance = balance

        self.list_IDs = self.make_list()
        self.maxindex = len(self.list_IDs)
        self.list_IDs_original = self.make_list()
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        nb = int(np.ceil(len(self.list_IDs) / self.batch_size))
        if nb == 0:
            raise ValueError('Batch size too large, number of batches per epoch is zero')
        return nb

    def __getitem__(self, index):
        """
        Generate one batch of data by:
            generating a list of indexes which corresponds to a list of ID, 
            use prepare_batch to prepare the batch
        """

        # print("Generating a new batch...")
        'Generate indexes of the batch'
        temp = (index + 1) * self.batch_size
        if temp < self.maxindex:
            indexes = self.indexes[index * self.batch_size:temp]
        else:
            indexes = self.indexes[index * self.batch_size:self.maxindex]

        'Find list of IDs'
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self.prepare_batch(self.list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        """
        Updates indexes and shuffle after each epoch
        """
        if self.balance:
            self.list_IDs = self.class_balancing()

        self.maxindex = len(self.list_IDs)
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def prepare_batch(self, list_IDs):
        """
        Prepare a batch of data:
            creating a list of images and masks after having preprocessed (and possibly augmented them)
            saving a few examples to disk if required
        """
        X = list()
        Y = list()
        S = list()

        xReadFunction = self.params['xReadFunction']
        yReadFunction = self.params['yReadFunction']

        for ID in list_IDs:
            if not isinstance(ID, np.ndarray):    
                ID = ID.replace("]", '').replace("[", '')
                ID = ID.replace("'", '').split(", ")
                ID[1] = int(ID[1])
                ID = np.asarray(ID)
            if len(self.params["distance"])<3:
                x, seg = xReadFunction(ID, self.params, im_mask="both", data=self.params["data"])
            else : 
                x, seg, ID[1] = xReadFunction(ID, self.params, im_mask="both", data=self.params["data"])

            x, seg = self.imaugment(x, seg)
            x = np.concatenate([x, seg], axis=-1)
            y = yReadFunction(ID, self.params)
            X.append(x)
            Y.append(y)

        X = np.asarray(X)
        Y = np.asarray(Y)
        
        return X, Y

    def imaugment(self, X, Y=None):
        """
        Preprocess the tuple (image,mask) and then apply if selected:
            augmentation techniques adapted from Keras ImageDataGenerator
            elastic deformation
        """

        if self.params["augmentation"][0] == True:
            X, Y = random_transform(X, Y, 
                **self.params["random_deform"])
                
        if self.params["augmentation"][1] == True:
            [X, Y] = elasticdeform.deform_random_grid([X, Y], 
                axis=[(0, 1), (0, 1)],
                sigma=self.params["e_deform_g"]["sigma"],
                points=self.params["e_deform_g"]["points"])
            
        if self.params["augmentation"][2] == True:
            r = np.random.normal(self.params["noise"]["mean"],
                self.params["noise"]["std"], X.shape)
            X = X + r.reshape(X.shape)
                
        if self.params["augmentation"][3]:
            g = 2 * self.params["gamma"]["range"][1]
            while ((g < self.params["gamma"]["range"][0]) | 
                (g > self.params["gamma"]["range"][1])):
                    g = np.random.normal(self.params["gamma"]["mean"],
                        self.params["gamma"]["std"])
            if self.params["norm"] is None:
                temp = (X - np.min(X)) / (np.max(X) - np.min(X))
                temp = np.power(temp, 1 / g)
                X = temp * (np.max(X) - np.min(X)) + np.min(X)
            else:
                X = 2 * np.power(X / 2 + 0.5, 1 / g) - 1
        return X, Y


    def save_images(self, X, Y, list_IDs, predict=False, overlay=False, epoch_number=None):
        """
        Save a png to disk (params["savefolder"]) to illustrate the data been generated
        predict: if set to True allows the saving of predicted images, remember to set Y and list_IDs as None
        the to_predict function can be used to reset the counter of saved images, this allows if shuffle is False to have the same order between saved generated samples and the predicted ones
        """
        if predict:
            genOrPred = "predict_"
        else:
            genOrPred = "generator_"

        if self.params["augmentation"] == [0, 0, 0]:
            augmentation = 'no_augm'
        else:
            augmentation = 'augm'

        if self.plotgenerator > self.plotedgenerator and (X[0].shape[-1] == 2 or X[0].shape[-1] >= 4 or (
                Y is not None and (Y[0] is not None and Y[0].shape[-1] == 2))):
            print("Not built to handle images with two channels or more than 3; try either 1 or 3")
        if self.plotgenerator > self.plotedgenerator:
            if len(X[0].shape) == 3:
                '''
                Save augmented images for 2D (will save 10 slices from different patients)
                '''
                nbr_samples = len(X)
                plt.figure(figsize=(6, 11), dpi=200)
                for i in range(min(nbr_samples, 10)):
                    im = X[i]
                    ax = plt.subplot(5, 2, i + 1)
                    plt.imshow(np.squeeze(im), cmap='gray', vmin=0, vmax=1)
                    plt.axis('off')
                    if predict:
                        pltname = "noname"
                    else:
                        pltname = list_IDs[i][-27:]
                    if not predict and (self.params["only"] == "both") and overlay:
                        plt.imshow(np.squeeze(Y[i]), cmap=plt.cm.Purples, alpha=.1)
                    fz = 5  # Works best after saving
                    ax.set_title(pltname, fontsize=fz)
                plt.savefig(os.path.join(self.params["savefolder"],
                                         str(self.params["data"]) + "_" + str(augmentation) +
                                         "_" + genOrPred + str(self.plotedgenerator) + "_ep" +
                                         str(epoch_number) + '_im.png'))
                plt.close()

            if not predict and (self.params["only"] == "both") and len(Y[0].shape) == 3:
                nbr_samples = len(X)
                plt.figure(figsize=(6, 11), dpi=200)
                for i in range(min(nbr_samples, 10)):
                    im = Y[i]
                    ax = plt.subplot(5, 2, i + 1)
                    plt.imshow(np.squeeze(im), cmap='gray', vmin=0, vmax=1)
                    plt.axis('off')
                    pltname = list_IDs[i][-27:]
                    fz = 5  # Works best after saving
                    ax.set_title(pltname, fontsize=fz)
                plt.savefig(os.path.join(self.params["savefolder"],
                                         str(self.params["data"]) + "_" + str(augmentation) +
                                         genOrPred + "_" + str(self.plotedgenerator) + "_ep" +
                                         str(epoch_number) + '_mask.png'))
                plt.close()

            if len(X[0].shape) == 4:
                '''
                Save augmented images for 3D (will save 10 slices from a single volume)
                '''
                Xto_print = X[0]
                mean_slice = int(list_IDs[0][1])
                dz = self.params["distance"][2]
                plt.figure(figsize=(6, 11), dpi=400)
                print(list_IDs[0])
                if predict:
                    plt.suptitle("noname", fontsize=5)
                else:
                    plt.suptitle(list_IDs[0][0], fontsize=5)
                
                for i in range(2*dz):
                    n_slice = mean_slice -dz + i
                    im = Xto_print[:, :, i]
                    ax = plt.subplot(5, 2, i + 1)
                    plt.imshow(np.squeeze(im), cmap='gray', vmin=0, vmax=1)
                    if not predict and (self.params["only"] == "both") and overlay:
                        plt.contour(np.squeeze(Y[0][:, :, i]), colors="g")
                    plt.axis('off')
                    pltname = "slice " + str(n_slice)
                    fz = 5  # Works best after saving
                    ax.set_title(pltname, fontsize=fz)
                plt.savefig(os.path.join(self.params["savefolder"],
                                         str(self.params["data"]) + str(self.params["augmentation"]) +
                                         genOrPred + "_" + str(self.plotedgenerator) + "_ep" + str(
                                             epoch_number) + '_im.png'))
                plt.close()
            if not predict and (self.params["only"] == "both") and len(Y[0].shape) == 4:
                Yto_print = Y[0]
                for j in range(Yto_print.shape[-1]):
                    plt.figure(figsize=(6, 11), dpi=400)
                    plt.suptitle(list_IDs[0][0], fontsize=5)
                    for i in range(2*dz):
                        n_slice = mean_slice -dz + i
                        im = Yto_print[:, :, i, j]
                        ax = plt.subplot(5, 2, i + 1)
                        plt.imshow(np.squeeze(im), cmap='gray', vmin=0, vmax=1)
                        plt.axis('off')
                        pltname = "slice " + str(n_slice)
                        fz = 5  # Works best after saving
                        ax.set_title(pltname, fontsize=fz)
                    plt.savefig(os.path.join(self.params["savefolder"],
                                             str(self.params["data"]) + str(self.params["augmentation"]) +
                                             genOrPred + "_" + str(self.plotedgenerator) + "_ep" + str(epoch_number) +
                                             '_mask_' + str(j) + '.png'))
                    plt.close()

            self.plotedgenerator += 1

    def to_predict(self):
        self.plotedgenerator = 0

    def class_balancing(self):
        """
        Repeat elements from smaller class until the class are balanced
        """
        patient_list = self.list_IDs_original
        for ID in self.list_IDs_original:
            labels = yReadFunction(patient_list, self.params)
            malignant = patient_list[labels == 1]
            tot_num = len(labels)
            mal_num = int(np.sum(labels))
            imbal = ((tot_num - mal_num) / mal_num) - 1
            if imbal >= 1:
                r = np.floor(imbal).astype(int)
                for j in range(r):
                    patient_list = np.append(patient_list, 
                        malignant, axis=0)
                imbal = imbal - r

            r = int(imbal * mal_num)
            r = random.sample(range(mal_num), r)
            patient_list = np.append(patient_list, malignant[r], 
                axis=0)
            return patient_list
            
            
    def make_list(self):
        """
        From partition, get minimum and maximum slice index
        """
        patient_list = np.empty(shape=(0, 3))
        
        for ID in self.list_IDs_original:
            ID = ID.replace("]", '').replace("[", '')
            ID = ID.replace("'", '').split(" ")
            ID = np.asarray(ID)
            patient_list = np.append(patient_list, [[ID[0], int(ID[1]), int(ID[-1])]], 
                axis = 0)
        return patient_list
