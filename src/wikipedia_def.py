from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pp.edward as ed  # Importing my own version of Edward
import numpy as np
import os
import tensorflow as tf

from datetime import datetime
from pp.edward.models import Gamma, Poisson, Normal, PointMass, \
    TransformedDistribution
from pp.edward.util import Progbar
from observations import nips
from optparse import OptionParser

ed.set_seed(42)

parser = OptionParser()
parser.add_option("-t", "--truncate", dest="truncate", default=False, action="store_true")
parser.add_option("-n", "--n_epoch", dest="n_epoch", default=40)
parser.add_option("-l", "--lr", dest="lr", default=0.2)
parser.add_option("-p", "--n_iter_per_epoch", dest="n_iter_per_epoch", default=100)
parser.add_option("-q", "--q", dest="q", default='gamma')
parser.add_option("-c", "--cv", dest="cv", default=False, action="store_true")
parser.add_option("-m", "--map", dest="map_estimate", default=False, action="store_true")
parser.add_option("-b", "--batch_size", dest="batch_size", default=100)
(options, args) = parser.parse_args()
truncate = options.truncate
n_epoch = int(options.n_epoch)
truncate = bool(options.truncate)
lr = float(options.lr)
n_iter_per_epoch = int(options.n_iter_per_epoch)
q = str(options.q)
cv = bool(options.cv)
map_estimate = bool(options.map_estimate)
M = int(options.batch_size) # minibatch size

logdir = '~/log/def/'
logdir = os.path.expanduser(logdir)

# DATA
with open('data/wikipedia_vocab.dat') as f:
  words = [line.rstrip() for line in f]
x_train = np.load('data/wikipedia_matrix.npy')

#M = 100 # minibatch size
N = x_train.shape[0]  # number of documents
D = x_train.shape[1]  # vocabulary size
K = [50, 25, 10]#, 30, 15]  # number of components per layer
shape = 0.1  # gamma shape parameter

from tensorflow.python.ops import random_ops
def _sample_n(self, n, seed=None):
    return tf.maximum(random_ops.random_gamma(
        shape=[n],
        alpha=self.concentration,
        beta=self.rate,
        dtype=self.dtype,
        seed=seed), 1e-300)

if truncate:
    print("Truncating Samples")
    Gamma._sample_n = _sample_n

# MODEL
Ws = {}
Ws['W0'] = Gamma(0.1, 0.3, sample_shape=[K[0], D])
for i in range(1, len(K)):
  Ws['W' + str(i)] = Gamma(0.1, 0.3, sample_shape=[K[i], K[i-1]])

zs = {}
zs['z' + str(len(K))] = Gamma(0.1, 0.1, sample_shape=[M, K[-1]])
for i in range(len(K) - 1, -1, -1):
  zs['z' + str(i)] = Gamma(shape, shape / tf.matmul(zs['z' + str(i + 1)], 
            Ws['W' + str(i)]))

x = Poisson(tf.matmul(zs['z1'], Ws['W0']))

# INFERENCE
def pointmass_q(shape):
  min_mean = 1e-3
  mean_init = tf.random_normal(shape)
  rv = PointMass(tf.maximum(tf.nn.softplus(tf.Variable(mean_init)), min_mean))
  return rv


def gamma_q(shape):
  # Parameterize Gamma q's via shape and scale, with softplus unconstraints.
  min_shape = 1e-3
  min_scale = 1e-5
  shape_init = -1.5 + 0.1 * tf.random_normal(shape)
  scale_init = 0.1 * tf.random_normal(shape)
  rv = Gamma(tf.maximum(tf.nn.softplus(tf.Variable(shape_init)),
                        min_shape),
             tf.maximum(1.0 / tf.nn.softplus(tf.Variable(scale_init)),
                        1.0 / min_scale))
  return rv

def next_batch(M):
  idx_batch = np.random.choice(len(x_train), M, replace=False)
  return x_train[idx_batch], idx_batch


min_shape = 0.005 # from DEF code
min_scale = 1e-5 # from DEF code

qWs = {}
if map_estimate:
  for i in range(len(K)):
    qWs['qW' + str(i)] = pointmass_q(Ws['W' + str(i)].shape)
else: 
  qW_variables = {}
  for i in range(len(K)):
    qW_variables['shape' + str(i)] = tf.Variable(0.5 + 0.1 * tf.random_normal(Ws['W' + str(i)].shape))
    qW_variables['scale' + str(i)] = tf.Variable(0.1 * tf.random_normal(Ws['W' + str(i)].shape))
    qWs['qW' + str(i)] = Gamma(tf.maximum(tf.nn.softplus(qW_variables['shape' + str(i)]), min_shape),
              tf.maximum(1.0 / tf.nn.softplus(qW_variables['scale' + str(i)]), 1.0 / min_scale))

idx_ph = tf.placeholder(tf.int32, M)
qz_variables = {}
qzs = {}
for i in range(1,len(K) + 1):
  qz_variables['shape' + str(i)] = tf.Variable(0.5 + 0.1 * tf.random_normal([N, K[i-1]]))
  qz_variables['scale' + str(i)] = tf.Variable(0.1 * tf.random_normal([N, K[i-1]]))
  qzs['qz' + str(i)] = Gamma(tf.maximum(tf.nn.softplus(tf.gather(qz_variables['shape' + str(i)], idx_ph)), min_shape),
          tf.maximum(1.0 / tf.nn.softplus(tf.gather(qz_variables['scale' + str(i)], idx_ph)), 1.0 / min_scale)) 

x_ph = tf.placeholder(tf.float32, [M, D]) 

inference_w_map = {}
inference_w_data = {}
inference_z_map = {}
inference_z_data = {}
scale_map = {}

inference_w_data[x] = x_ph
inference_z_data[x] = x_ph
scale_map[x] = float(N) / M

for i in range(len(K)):
  inference_w_map[Ws['W' + str(i)]] = qWs['qW' + str(i)]
  inference_w_data[zs['z' + str(i + 1)]] = qzs['qz' + str(i + 1)]
  
  inference_z_map[zs['z' + str(i + 1)]] = qzs['qz' + str(i + 1)]
  inference_z_data[Ws['W' + str(i)]] = qWs['qW' + str(i)]

  scale_map['z' + str(i + 1)] = float(N) / M

if map_estimate:
  inference_w = ed.MAP(inference_w_map, data=inference_w_data)
else:
  inference_w = ed.KLqp(inference_w_map, data=inference_w_data)
inference_z = ed.KLqp(inference_z_map, data=inference_z_data)

#optimizer_w = tf.train.RMSPropOptimizer(lr)
#optimizer_z = tf.train.RMSPropOptimizer(lr)
timestamp = datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S")
logdir += timestamp + '_' + '_'.join([str(ks) for ks in K]) + \
    '_q_' + str(q) + '_lr_' + str(lr)

if map_estimate:
  inference_w.initialize(scale=scale_map)
else:
  inference_w.initialize(scale=scale_map,
                         control_variates=cv,
                         n_samples=64)
inference_z.initialize(scale=scale_map,
                       control_variates=cv,
                       logdir=logdir,
                       n_samples=64)

sess = ed.get_session()
tf.global_variables_initializer().run()

print(N, "documents")
print(D, "tokens")
print("Log directory: ", logdir)
for epoch in range(n_epoch):
  print("Epoch {}".format(epoch))
  nll = 0.0

  pbar = Progbar(n_iter_per_epoch)
  for t in range(1, n_iter_per_epoch + 1):
    x_batch, idx_batch = next_batch(M)
    pbar.update(t)
    info_dict_e = inference_z.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch})
    info_dict_m = inference_w.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch})
    nll += info_dict_e['loss']

  # Compute perplexity averaged over a number of training iterations.
  # The model's negative log-likelihood of data is upper bounded by
  # the variational objective.
  nll = nll / n_iter_per_epoch
  perplexity = np.exp(nll / np.sum(x_batch)) # or should this be x_train?
  print("Negative log-likelihood <= {:1.3f}".format(nll))
  print("Perplexity <= {:0.3f}".format(perplexity))

  # Print top 10 words for first 10 topics.
  qW0_vals = sess.run(qWs['qW0'])
  for k in range(10):
    top_words_idx = qW0_vals[k, :].argsort()[-10:][::-1]
    top_words = " ".join([words[i] for i in top_words_idx])
    print("Topic {}: {}".format(k, top_words))

print("Log directory: ", logdir)

