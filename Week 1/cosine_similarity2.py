#finding cosine similarity of 2 sentences

from math import sqrt

def sent_to_vect(sentence,all_words) :
     return [sentence.count(words) for words in all_words]

s1 = ("Hello my world")
s2 = ("Hello my world world")

s1_words= s1.lower().split()
s2_words= s2.lower().split()

all_words= list(set(s1_words+s2_words))

v1= sent_to_vect(s1_words,all_words)
v2= sent_to_vect(s2_words,all_words)

sum_a = 0
sum_b = 0
sum_both = 0

for i,j in zip(v1,v2):
  sum_a = sum_a + i*i
  sum_b = sum_b + j*j
  sum_both = sum_both + i*j

ans = sum_both/ (sqrt(sum_a) * sqrt(sum_b))
print("Cosine Similarity of two sentences are:",ans)


          