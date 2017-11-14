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
(options, args) = parser.parse_args()
truncate = options.truncate
n_epoch = int(options.n_epoch)
truncate = bool(options.truncate)
lr = float(options.lr)
n_iter_per_epoch = int(options.n_iter_per_epoch)
q = str(options.q)
cv = bool(options.cv)
map_estimate = bool(options.map_estimate)

data_dir = "~/data"
logdir = '~/log/def/'
data_dir = os.path.expanduser(data_dir)
logdir = os.path.expanduser(logdir)

# DATA
x_train, metadata = nips(data_dir)
documents = metadata['columns']
words = metadata['rows']

# Subset to documents in 2011 and words appearing in at least two
# documents and have a total word count of at least 10.
doc_idx = [i for i, document in enumerate(documents)
           if document.startswith('2011')]
documents = [documents[doc] for doc in doc_idx]
x_train = x_train[:, doc_idx]
word_idx = np.logical_and(np.sum(x_train != 0, 1) >= 2,
                          np.sum(x_train, 1) >= 10)
words = [word for word, idx in zip(words, word_idx) if idx]
x_train = x_train[word_idx, :]
x_train = x_train.T

M = 100 # minibatch size
N = x_train.shape[0]  # number of documents
D = x_train.shape[1]  # vocabulary size
#K = [100]
K = [100, 30, 15]  # number of components per layer
shape = 0.1  # gamma shape parameter

from tensorflow.python.ops import random_ops
def _sample_n(self, n, seed=None):
    return tf.maximum(random_ops.random_gamma(
        shape=[n],
        alpha=self.concentration,
        beta=self.rate,
        dtype=self.dtype,
        seed=seed), 1e-8)

if truncate:
    print("Truncating Samples")
    Gamma._sample_n = _sample_n

# MODEL
W2 = Gamma(0.1, 0.3, sample_shape=[K[2], K[1]])
W1 = Gamma(0.1, 0.3, sample_shape=[K[1], K[0]])
W0 = Gamma(0.1, 0.3, sample_shape=[K[0], D])

z3 = Gamma(0.1, 0.1, sample_shape=[M, K[2]]) # Changed to minibatch
z2 = Gamma(shape, shape / tf.matmul(z3, W2))
z1 = Gamma(shape, shape / tf.matmul(z2, W1))
#z1 = Gamma(0.1, 0.1, sample_shape=[M, K[0]])
x = Poisson(tf.matmul(z1, W0))


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
  idx_batch = np.random.choice(len(x_train), M)
  return x_train[idx_batch], idx_batch


min_shape = 1e-3
min_scale = 1e-5

if map_estimate:
  qW2 = pointmass_q(W2.shape)
  qW1 = pointmass_q(W1.shape)
  qW0 = pointmass_q(W0.shape)
else: 
  qW0_variables = [tf.Variable(0.5 + 0.1 * tf.random_normal(W0.shape)),
                  tf.Variable(0.1 * tf.random_normal(W0.shape)),
                  tf.Variable(0.5 + 0.1 * tf.random_normal(W1.shape)),
                  tf.Variable(0.1 * tf.random_normal(W1.shape)),
                  tf.Variable(0.5 + 0.1 * tf.random_normal(W2.shape)),
                  tf.Variable(0.1 * tf.random_normal(W2.shape))]

  qW0 = Gamma(tf.maximum(tf.nn.softplus(qW0_variables[0]), min_shape),
              tf.maximum(1.0 / tf.nn.softplus(qW0_variables[1]), 1.0 / min_scale))
  qW1 = Gamma(tf.maximum(tf.nn.softplus(qW0_variables[2]), min_shape),
              tf.maximum(1.0 / tf.nn.softplus(qW0_variables[3]), 1.0 / min_scale))
  qW2 = Gamma(tf.maximum(tf.nn.softplus(qW0_variables[4]), min_shape),
              tf.maximum(1.0 / tf.nn.softplus(qW0_variables[5]), 1.0 / min_scale))

qz_variables = [tf.Variable(0.5 + 0.1 * tf.random_normal([N, K[0]])), 
              tf.Variable(0.1 * tf.random_normal([N, K[0]])),
              tf.Variable(0.5 + 0.1 * tf.random_normal([N, K[1]])), 
              tf.Variable(0.1 * tf.random_normal([N, K[1]])),
              tf.Variable(0.5 + 0.1 * tf.random_normal([N, K[2]])), 
              tf.Variable(0.1 * tf.random_normal([N, K[2]]))]

idx_ph = tf.placeholder(tf.int32, M)
qz1 = Gamma(tf.maximum(tf.nn.softplus(tf.gather(qz_variables[0], idx_ph)), min_shape),
          tf.maximum(1.0 / tf.nn.softplus(tf.gather(qz_variables[1], idx_ph)), 1.0 / min_scale)) 
qz2 = Gamma(tf.maximum(tf.nn.softplus(tf.gather(qz_variables[2], idx_ph)), min_shape),
          tf.maximum(1.0 / tf.nn.softplus(tf.gather(qz_variables[3], idx_ph)), 1.0 / min_scale))
qz3 = Gamma(tf.maximum(tf.nn.softplus(tf.gather(qz_variables[4], idx_ph)), min_shape),
          tf.maximum(1.0 / tf.nn.softplus(tf.gather(qz_variables[5], idx_ph)), 1.0 / min_scale))
x_ph = tf.placeholder(tf.float32, [M, D]) 
# We apply variational EM with E-step over local variables
# and M-step to point estimate the global weight matrices.
#inference_e = ed.KLqp({z1: qz1},#, z2: qz2, z3: qz3},
#                      data={x: x_ph, W0: qW0})#, W1: qW1, W2: qW2})
#inference_m = ed.MAP({W0: qW0},#, W1: qW1, W2: qW2},
#                     data={x: x_ph, z1: qz1})#, z2: qz2, z3: qz3})

if map_estimate:
  inference_w = ed.MAP({W0: qW0, W1: qW1, W2: qW2},
                       data={x: x_ph, z1: qz1, z2: qz2, z3: qz3})
else:
  inference_w = ed.KLqp({W0: qW0, W1: qW1, W2: qW2},
                       data={x: x_ph, z1: qz1, z2: qz2, z3: qz3})
inference_z = ed.KLqp({z1: qz1, z2: qz2, z3: qz3},
                       data={x: x_ph, W0: qW0, W1: qW1, W2: qW2})

#optimizer_w = tf.train.RMSPropOptimizer(lr)
#optimizer_z = tf.train.RMSPropOptimizer(lr)
timestamp = datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S")
logdir += timestamp + '_' + '_'.join([str(ks) for ks in K]) + \
    '_q_' + str(q) + '_lr_' + str(lr)

if map_estimate:
  inference_w.initialize(scale={x: float(N) / M, z1: float(N) / M})
else:
  inference_w.initialize(scale={x: float(N) / M, z1: float(N) / M},
                         control_variates=cv)
                       #optimizer=optimizer_w)
                       #var_list = qz_variables,
                       #n_samples = 5,
                       #control_variates=False)
inference_z.initialize(scale={x: float(N) / M, z1: float(N) / M},
                       #optimizer=optimizer_z,
                       #var_list = qW0_variables,
                       control_variates=False,
                       n_samples=5,
                       logdir=logdir)

#kwargs = {'optimizer': optimizer_e,
#          'n_print': 100,
#          'logdir': logdir,
#          'scale': {x: float(N) / M, z1: float(N) / M},
#          #'scale': {x: float(N) / M, z1: float(N) / M, 
#          #                    z2: float(N) / M, z3: float(N) / M},      
#          'log_timestamp': False,
#          'var_list': qz_variables}
#
#if q == 'gamma':
#  kwargs['n_samples'] = 30
#if cv:
#    print("Using control variates")
#    kwargs['control_variates'] = True
#else:
#    print("Not using control variates")
#    kwargs['control_variates'] = False
#inference_e.initialize(**kwargs)
#inference_m.initialize(optimizer=optimizer_m)

sess = ed.get_session()
tf.global_variables_initializer().run()

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
    #info_dict_e = inference_e.update()
    #info_dict_m = inference_m.update()
    nll += info_dict_e['loss']

  # Compute perplexity averaged over a number of training iterations.
  # The model's negative log-likelihood of data is upper bounded by
  # the variational objective.
  nll = nll / n_iter_per_epoch
  perplexity = np.exp(nll / np.sum(x_train))
  print("Negative log-likelihood <= {:1.3f}".format(nll))
  print("Perplexity <= {:0.3f}".format(perplexity))

  # Print top 10 words for first 10 topics.
  qW0_vals = sess.run(qW0)
  for k in range(10):
    top_words_idx = qW0_vals[k, :].argsort()[-10:][::-1]
    top_words = " ".join([words[i] for i in top_words_idx])
    print("Topic {}: {}".format(k, top_words))

print("Log directory: ", logdir)

