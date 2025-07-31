#finding cosine similarity of 2 numbers 

from math import sqrt

a = [2,3,0,1]
b = [2,8,4,0,3]

sum_a = 0
sum_b = 0
sum_both = 0

for i,j in zip(a,b):
  sum_a = sum_a + i*i
  sum_b = sum_b + j*j
  sum_both = sum_both + i*j

ans = sum_both/ (sqrt(sum_a) * sqrt(sum_b))
print(ans)