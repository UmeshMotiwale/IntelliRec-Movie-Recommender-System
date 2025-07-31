import numpy as np

#assuminh two users rating two movies, usersxratings
M = np.array([[5,3],
              [2,1]])

#Guessing the predicated matrix using N latent features
N=5 #No. of latent features

P= np.random.rand(2,N)
Q= np.random.rand(2,N)

M_predic = np.dot(P,Q.T)

print("Original matrix: " ,M)
print("Predicted Matrix: ",M_predic)