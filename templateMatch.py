import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as patches
import math as mt
import time


class templateMatch(object):

    def __init__(self,baseImage,template,downsample):
        self.baseImage = plt.imread(baseImage).mean(axis=2)
        self.template = plt.imread(template).mean(axis=2)
        self.downsample = downsample
        self.matches = []

    
    def gauss_kernal(self,size, var):
        kernel = np.zeros(shape=(size,size))
        for i in range(size):
            for j in range(size):
                kernel[i][j] = mt.exp( -((i - (size-1)/2)**2 ))
        kernel = kernel / kernel.sum()
        return kernel
                                     
    def convolve(self,g,h): # h is kernel, g is the image
        
        I_gray_copy = np.zeros(shape=(len(g[:,1]),len(g[1,:])))
        u,v = h.shape
        sp = int(u/2)

        start = sp
        for i in range(start,len(g[:,1])-1):
            for j in range(start, len(g[i,:])-1):
                f = g[i-sp:i+sp, j-sp:j+sp] #FIXME
                # need another array to put value into
                total = h*f
                I_gray_copy[i][j] = sum(sum(total))
        return I_gray_copy

    def downSample(self,image,n=None):
        if n is None:
            n = int(self.downsample)
        pyramid = []
        for i in range(n):
            image_conv = self.convolve(image,self.gauss_kernal(4,1))
            pyramid.append(image_conv)
            image = image[::2,::2]
        return pyramid

    def templMatch(self,BaseImage,matchImage):
        TA = time.time()
        U,V = BaseImage.shape
        pyramid = self.downSample(matchImage)
        #count = 0 #FIXME
        #print("U,V : ",U,V)
        for i in range(len(pyramid)):
            subMatchImage = pyramid[i]
            mU,mV = subMatchImage.shape
            argmin = np.inf
            bestU,bestV = None,None
            #print("Cross corr test start\n")
            for j in range(0,U-mU):
                for k in range(0,V-mV):
                
                   # compare = self.SSE(BaseImage[j:mU+j,k:mV+k],matchImage)
                    compare = ((BaseImage[j:mU+j,k:mV+k] - subMatchImage)**2).sum()/(mU*mV)
                
                    if(compare < argmin):
                       #count += 1
                       #print("Better match found,count: ",count)
                       argmin = compare
                       bestU = j
                       bestV = k
                #print("j loop: ",j)
            self.matches.append((i+1,bestU,bestV,argmin))
            #print("Cross corr test end\n")
        self.matches.sort(key = lambda x:x[-1])
        print("Best location estimate: ", self.matches[0][1],self.matches[0][2])
        print("All estimates: \n")

        for i in range(int(self.downsample)):
            print(self.matches[i][0] ," " ,self.matches[i][1], " ", self.matches[i][2]," ",self.matches[i][3])

        print("Elapsed Time: ",time.time()-TA, "\n")              
        return self.matches

    def plotBestMatch(self,matches,baseImage,template):
        lev = matches[0][0]
        tx,ty = template.shape
        tx,ty = tx/lev, ty/lev
        bx,by = baseImage.shape
        X,Y = lev*matches[0][1], lev*matches[0][2] #location of best match in base image
        #coords for square
        cX,cY = X+(tx/2),Y+(ty/2) #center of square
        #p1,p2,p3,p4 = [X,Y], [X + tx/2,Y], [X + tx/2,Y + ty/2], [X ,Y + ty/2]
        fig,ax = plt.subplots(1)
        ax.imshow(self.baseImage)
        rect = patches.Rectangle([self.matches[0][1],self.matches[0][2]],ty,tx,linewidth=5,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
        plt.show()
        #plt.savefig("WheresWaldo.jpg")

    def runAndPlot(self):
        match = self.templMatch(self.baseImage,self.template)
        self.plotBestMatch(match,self.baseImage,self.template)



if __name__ == "__main__":
  WheresWaldo = templateMatch(sys.argv[1],sys.argv[2],sys.argv[3])
  matches = WheresWaldo.runAndPlot()



