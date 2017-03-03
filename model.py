import numpy as np
from read_image import asRowMatrix
from read_image import pca, project
from distance import EuclideanDistance
from read_image import read_images
class BaseModel(object):
    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):
        self.dist_metric= dist_metric
        self.num_components=0
        self.projections=[]
        self.mu=[]
        self.W=[]
        if(X is not None) and (y is not None):
            self.compute(self,X, y)
    def compute(self, X, y):
        raise NotImplementedError("Every BaseModel must implement the compute method.")
    def predict(self, X):
        minDist=np.finfo('float').max
        minClass=-1
        Q=project(self.W, X.reshape(1,-1), self.mu)
        for i in xrange(len(self.projections)):
            dist=self.dist_metric(self.projections[i], Q)
            if dist<minDist:
                minClass=self.y[i]
                minDist= dist
        return minClass
class EigenfaceModel(BaseModel):
    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):
        super(EigenfaceModel, self).__init__(X=X, y=y, dist_metrix=dist_metric,
                                             num_components=num_components)
    def compute(self, X, y):
        [D, self.W, self.mu]=pca(asRowMatrix(X), y, self.num_components)
        # store labels
        self.y=y
        #store projections
        for xi in X:
            self.projections.append(project(self.reshape(1,-1), self.mu))
###Now that the eigenfacesModel is defined, it can be used to learn the Eigenfaces and generate predictions

##Example
import sys
sys.path.append("..")
import numpy as np
path="C:\Users\\12815\\Desktop\\face\\Training Set"
[X, y]=read_images(path)
model=EigenfaceModel(X[1:], y[1:])
print "expected=", y[0], "/", "predicted= ", model.predict(X[0])
