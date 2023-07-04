
import numpy as np 
import os 
import sys 
from PIL import Image  # pip install pillow 
import matplotlib.cm as cm # pip install matplotlib 
import matplotlib.pyplot as plt  
from DistanceMetric import * 


def normalize(X, low, high , dtype=None):  
        X = np.asarray(X)  
        minX , maxX = np.min(X), np.max(X)  
        # normalize to [0...1].  
        X = X - float(minX)  
        X = X / float((maxX - minX)) # scale to [low...high].  
        X = X * (high -low)  
        X = X + low  
        if dtype is None:  
            return np.asarray(X)  
        return np.asarray(X, dtype=dtype) 
 
def read_images(path , sz=None):  
    c = 0  
    X,y,yseq = [], [], [] # list of images and labels, yseq is sequential ordering number 
    for dirname , dirnames , filenames in os.walk(path):  
        for filename in filenames:  
            try: 
                fname = path+filename 
                im = Image.open(fname)  
                im = im.convert("L") # resize to given size (if given) 
                if (sz is not None):  
                    im = im.resize(sz, Image.ANTIALIAS) 
                imdata = np.asarray(im, dtype=np.uint8) 
                sh = imdata.shape 
                X.append(np.asarray(im, dtype=np.uint8))  
                sh = len(X) 
                label = filename.split('_',1)[0]  # e.g. S10 
                y.append(label)  # use this to determine accuracy 
                yseq.append(c) 
                #y.append(filename)  uncomment this to show individual test 
                c = c + 1 
            except IOError:  
                print("I/O error({0}): {1}".format(errno , strerror))  
    return [X,y,yseq] 
 
def create_font(fontname='Tahoma', fontsize=10):  
        return { 'fontname': fontname , 'fontsize':fontsize } 

def subplot(title , images , rows , cols , sptitle="subplot", sptitles=[], 
colormap=cm.gray , ticks_visible=True , filename=None):  
    fig = plt.figure()  
    # main title  
    fig.text(.5, .95, title , horizontalalignment='center')  
    for i in range(len(images)):  
        ax0 = fig.add_subplot(rows ,cols ,(i+1))  
        plt.setp(ax0.get_xticklabels(), visible=False)  
        plt.setp(ax0.get_yticklabels(), visible=False) 
        if len(sptitles) == len(images):  
            plt.title("%s #%s" % (sptitle , str(sptitles[i])), create_font('Tahoma',10))  
        else:  
            plt.title("%s #%d" % (sptitle , (i+1)), create_font('Tahoma',10))  
        plt.imshow(np.asarray(images[i]), cmap=colormap)  
    if filename is None:  
        plt.show()  
    else:  
        fig.savefig(filename) 






