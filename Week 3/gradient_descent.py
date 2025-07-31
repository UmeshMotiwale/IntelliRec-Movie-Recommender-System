import numpy as np

#assumingtwo users rating two movies, users x ratings
M = np.array([[5,3],
              [2,1],
              [4,3]])

N=2 #No. of latent features
n_users, n_items = M.shape #(2,2)

P= np.random.rand(n_users,N) #matrix od user and features
Q= np.random.rand(n_items,N) #matrix of items and features
    
#gradient descent for accounting error 
x=0.01    
y=0.01

for step in range(200):
    for i in range(n_users):
        for j in range(n_items):
            if M[i][j]>0:
                prediction= np.dot(P[i],Q[j].T)
                error = M[i][j] - prediction

                for k in range(N):
                    P[i][k] += x * (2 * error * Q[j][k] - y * P[i][k])
                    Q[j][k] += x * (2 * error * P[i][k] - y * Q[j][k])

        
M_predict  = np.dot(P,Q.T)

print("Original matrix: \n" ,M)
print("Predicted Matrix: \n",M_predict)