'''The header:

n_examples n_words
Followed by n_examples examples, each has two lines:

example_ind example_words
word_ind0 word_count0 word_ind1 word_count1 ...'''

import numpy as np

with open('data/train.cpp.dat') as f:
    lines = f.readlines()

N = int(lines[0].split(' ')[0])
D = int(lines[0].split(' ')[1])

x_train = np.zeros((N, D)).astype(int)

print(N, D)

for line_ind in range(2, len(lines), 2):
	document_ind = int(lines[line_ind-1].split(' ')[0])
	word_inds = np.array(list(map(int, lines[line_ind].split(' '))))
	for i in range(0, len(word_inds), 2):
		#print(word_inds[i+1])
		x_train[document_ind, word_inds[i]] = word_inds[i+1]

np.save('data/wikipedia_matrix', x_train)