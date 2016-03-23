# -*- coding: utf-8 -*-
import numpy as np



class ImPatch():
    def __init__(self, image_shape = (512, 384), n = 0, patch_width = 7, stride = 1):
        self.image_shape = image_shape
        self.n = n
        self.patch_width = patch_width
        self.stride = stride
        
        xlength = image_shape[0]
        ylength = image_shape[1]
        self.patch_size = patch_width**2*3  
        
        if patch_width > xlength or patch_width > ylength:
            raise Exception("Patchsize too big for given image")
    
    
        # Max top left index from which patches are taken        
        xindexmax = xlength - patch_width    
        yindexmax = ylength - patch_width 
        
        self.coords = [(x,y) for x in np.arange(0,xindexmax+1, stride) for y in np.arange(0, yindexmax+1, stride)]
        
        self.nmaxpatches = len(self.coords)
            
        
        if self.n > self.nmaxpatches:
            raise Exception("Impossible to extract this many patches from image")
            
        if n == 0:
            self.n = self.nmaxpatches
                    
    

    def patch(self, image):    
        """
           Patches an image (samples sub-images)
        """
        if(len(image.shape)==3):
            patches = np.zeros((self.n,self.patch_size))
        else:
            patches = np.zeros((self.n,self.patch_size/3))
        
        # Shuffle list of coords
        #random.shuffle(coords)
        
        
        for i, coord in enumerate(self.coords):
            if i >= self.n:
                break
            
            x, y = coord
            image = np.reshape(image, (-1,image.shape[-2], image.shape[-1]))
            patch = image[:,x:(x+self.patch_width),y:(y+self.patch_width)]
            np_patch = np.reshape(patch, -1)
            patches[i] = np_patch
        
        
        mid_coords = [(x+2,y+2) for (x,y) in self.coords]        
        
        return patches,mid_coords
        
        
    def npatch(self):
        return self.nmaxpatches
        """
            Maximum amount of patches extracted from image given size
        """
    

    
         
    