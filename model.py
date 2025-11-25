import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self,alpha=0.01,num_inters=1000):
        self.alpha=alpha
        self.num_inters=num_inters
        self.w=None
        self.b=None
        self.mean=None
        self.std=None
        self.J_history=[]

    def _scale_fit(self,x):
        self.mean=x.mean(axis=0)
        self.std=x.std(axis=0)
        return (x-self.mean)/self.std

    def _scale_transform(self,x):
        return(x-self.mean)/self.std

    def fit(self,X,y):
        X=np.array(X,dtype=float)
        y=np.array(y,dtype=float)

        X=self._scale_fit(X)
        m,n=X.shape
        self.w=np.zeros(n)
        self.b=0.0

        for i in range(self.num_inters):
            preds= X @ self.w + self.b
            error= preds - y

            dw=(1/m)*(X.T @ error)
            db=(1/m)*np.sum(error)
            
            self.w-=self.alpha*dw
            self.b-=self.alpha*db
            
            if i % 50==0:
                cost =(1/(2*m))*np.sum(error**2)
                self.J_history.append(cost)

    def predict(self,X):

        X=np.array(X,dtype=float)

        if  X.ndim==1:
            X=X.reshape(1,-1)
        X_scaled=self._scale_transform(X)
        return X_scaled @ self.w + self.b
                    
def train_example_model():
   """Example usage of LinearRegressionGD class
   """
   X = np.array([
       [500,1,2,35],
       [700,2,2,40],    
       [800,3,3,45],
       [900,4,3,50]
    
   ])

   y = np.array([150,200,250,300])
    
   model = LinearRegressionGD(alpha=0.0001,num_inters=5000)
   model.fit(X,y)
   return model
  
