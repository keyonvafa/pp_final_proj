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
from tensorflow.python.ops import random_ops


def load_nips_data():
    # Subset to  words appearing in at least two
    # documents and have a total word count of at least 10.
    # Note: Copied from Edward example
    data_dir = "~/data"
    data_dir = os.path.expanduser(data_dir)

    x_full, metadata = nips(data_dir)
    documents = metadata['columns']
    words = metadata['rows']

    doc_idx = [i for i, document in enumerate(documents)]
    documents = [documents[doc] for doc in doc_idx]
    x_full = x_full[:, doc_idx]
    word_idx = np.logical_and(
        np.sum(x_full != 0, 1) >= 2,
        np.sum(x_full, 1) >= 10
    )
    words = [word for word, idx in zip(words, word_idx) if idx]
    return x_full, words


def make_savedir(K, skip, q, map_estimate, lr, data='nips', most_skip=False):
        timestamp = datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S")
        savedir = 'info/'
        savedir += timestamp + '_' + data + '_' + '_'.join(
            [str(ks) for ks in K]
        )
        savedir += '_' + q
        savedir += '_lr_' + str(lr)
        if skip:
                savedir = savedir + '_skip'
        if map_estimate:
                savedir = savedir + '_map'
        if most_skip:
                savedir = savedir + '_most_skip'
        return savedir


def load_from_savedir(savedir, K):
        ws = [0 for i in range(len(K))]
        zs = [0 for i in range(len(K))]
        for i in range(len(K)):
                ws[i] = np.load(savedir + '/W' + str(i) + '.npy')
                zs[i] = np.load(savedir + '/z' + str(i + 1) + '.npy')
        losses = np.load(savedir + '/losses.npy')
        perps = np.load(savedir + '/test_perplexities.npy')
        return perps, losses, zs, ws


def train_and_test(
    architecture, savedir, skip_connections=False, q='lognormal',
    lr=0.01, n_epoch=1, n_iter_per_epoch=40000, cv=False, batch_size=-1,
    truncate=True, seed=42, n_test_epoch=50, n_test_iter=500,
    map_estimate=False, data='nips', most_skip=False
):

    ed.set_seed(seed)
    sess = tf.InteractiveSession()
    with sess.as_default():
        M = batch_size
        K = architecture

        logdir = '~/log/def/'
        logdir = os.path.expanduser(logdir)

        # DATA
        if data == 'nips':
            data_dir = "~/data"
            data_dir = os.path.expanduser(data_dir)
            x_full, metadata = nips(data_dir)
            documents = metadata['columns']
            words = metadata['rows']

            # Subset to documents in 2011 and words appearing in at least two
            # documents and have a total word count of at least 10.
            doc_idx = [i for i, document in enumerate(documents)]
            documents = [documents[doc] for doc in doc_idx]
            x_full = x_full[:, doc_idx]
            word_idx = np.logical_and(
                np.sum(x_full != 0, 1) >= 2,
                np.sum(x_full, 1) >= 10
            )
            words = [word for word, idx in zip(words, word_idx) if idx]
            x_full = x_full[word_idx, :]
            x_full = x_full.T
        elif data == 'ap':
            words = np.load('data/ap_words.npy')
            x_full = np.load('data/ap_data.npy')
        else:  # otherwise it's 20 newsgroups
            words = np.load('data/20_newsgroups_words.npy')
            x_full = np.load('data/20_newsgroups_data.npy')

        train_inds = np.random.choice(
            len(x_full),
            (len(x_full) - 1000),
            replace=False
        )
        test_inds = np.setdiff1d(np.arange(len(x_full)), train_inds)

        x_train = x_full[train_inds]
        x_test = x_full[test_inds]

        N = x_train.shape[0]  # number of documents
        D = x_train.shape[1]  # vocabulary size
        shape = 0.1  # gamma shape parameter

        def _sample_n(self, n, seed=None):
            return tf.maximum(random_ops.random_gamma(
                shape=[n],
                alpha=self.concentration,
                beta=self.rate,
                dtype=self.dtype,
                seed=seed), 1e-8
            )

        if truncate:
                Gamma._sample_n = _sample_n

        # MODEL
        Ws = {}
        Ws['W0'] = Gamma(0.1, 0.3, sample_shape=[K[0], D])
        for i in range(1, len(K)):
            Ws['W' + str(i)] = Gamma(0.1, 0.3, sample_shape=[K[i], K[i - 1]])

        zs = {}
        if M == -1:
            zs['z' + str(len(K))] = Gamma(0.1, 0.1, sample_shape=[N, K[-1]])
        else:
            zs['z' + str(len(K))] = Gamma(0.1, 0.1, sample_shape=[M, K[-1]])
        for i in range(len(K) - 1, 0, -1):
            scale_denominator = tf.matmul(
                zs['z' + str(i + 1)],
                Ws['W' + str(i)]
            )
            if skip_connections & (i % 3 == 1):
                scale_denominator = scale_denominator + zs['z' + str(i + 2)]
            if most_skip:
                    if i % 3 == 1:
                        scale_denominator = scale_denominator + zs[
                            'z' + str(i + 2)
                        ] + zs[
                            'z' + str(i + 1)
                        ]
                    if i % 3 == 2:
                        scale_denominator = scale_denominator + zs[
                            'z' + str(i + 1)
                        ]
            zs['z' + str(i)] = Gamma(shape, shape / scale_denominator)

        x = Poisson(tf.matmul(zs['z1'], Ws['W0']))

        def next_batch(M):
            idx_batch = np.random.choice(len(x_train), M, replace=False)
            return x_train[idx_batch], idx_batch

        min_shape = 1e-3
        min_scale = 1e-5
        min_mean = 1e-3

        qWs = {}
        qW_variables = {}

        if M != -1:
            idx_ph = tf.placeholder(tf.int32, M)
        qz_variables = {}
        qzs = {}
        for i in range(1, len(K) + 1):
            if map_estimate:
                qW_variables[str(i - 1)] = tf.Variable(
                    tf.random_normal(
                        Ws['W' + str(i - 1)].shape
                    ),
                    name='W' + str(i)
                )
                qWs['qW' + str(i - 1)] = PointMass(
                    tf.maximum(
                        tf.nn.softplus(
                            qW_variables[str(i - 1)]),
                        min_mean
                    )
                )
                if q == 'lognormal':
                    qz_variables['loc' + str(i)] = tf.Variable(
                        tf.random_normal([N, K[i - 1]]),
                        name='loc_z' + str(i)
                    )
                    qz_variables['scale' + str(i)] = tf.Variable(
                        0.1 * tf.random_normal([N, K[i - 1]]),
                        name='scale_z' + str(i)
                    )
                    if M != -1:
                        qzs['qz' + str(i)] = TransformedDistribution(
                            distribution=Normal(
                                tf.gather(
                                    qz_variables['loc' + str(i)],
                                    idx_ph
                                ),
                                tf.maximum(
                                    tf.nn.softplus(
                                        tf.gather(
                                            qz_variables['scale' + str(i)],
                                            idx_ph
                                        )
                                    ),
                                    min_scale
                                )
                            ),
                            bijector=tf.contrib.distributions.bijectors.Exp())
                    else:
                        qzs['qz' + str(i)] = TransformedDistribution(
                            distribution=Normal(
                                qz_variables['loc' + str(i)],
                                tf.maximum(
                                    tf.nn.softplus(
                                        qz_variables['scale' + str(i)]
                                    ),
                                    min_scale
                                )
                            ),
                            bijector=tf.contrib.distributions.bijectors.Exp()
                        )
                else:
                    qz_variables['shape' + str(i)] = tf.Variable(
                        0.5 + 0.1 * tf.random_normal([N, K[i - 1]])
                    )
                    qz_variables['scale' + str(i)] = tf.Variable(
                        0.1 * tf.random_normal([N, K[i - 1]])
                    )
                    if M != -1:
                        qzs['qz' + str(i)] = Gamma(
                            tf.maximum(
                                tf.nn.softplus(
                                    tf.gather(
                                        qz_variables['shape' + str(i)],
                                        idx_ph
                                    )
                                ),
                                min_shape
                            ),
                            tf.maximum(
                                1.0 / tf.nn.softplus(
                                    tf.gather(
                                        qz_variables['scale' + str(i)],
                                        idx_ph
                                    )
                                ),
                                1.0 / min_scale
                            )
                        )
                    else:
                        qzs['qz' + str(i)] = Gamma(
                            tf.maximum(
                                tf.nn.softplus(
                                    qz_variables['shape' + str(i)]
                                ),
                                min_shape
                            ),
                            tf.maximum(
                                1.0 / tf.nn.softplus(
                                    qz_variables['scale' + str(i)]
                                ),
                                1.0 / min_scale
                            )
                        )
            else:
                if q == 'lognormal':
                    qW_variables['loc' + str(i - 1)] = tf.Variable(
                        tf.random_normal(Ws['W' + str(i - 1)].shape),
                        name='loc_w' + str(i - 1)
                    )
                    qW_variables['scale' + str(i - 1)] = tf.Variable(
                        0.1 * tf.random_normal(Ws['W' + str(i - 1)].shape),
                        name='scale_w' + str(i - 1)
                    )
                    qz_variables['loc' + str(i)] = tf.Variable(
                        tf.random_normal([N, K[i - 1]]),
                        name='loc_z' + str(i)
                    )
                    qz_variables['scale' + str(i)] = tf.Variable(
                        0.1 * tf.random_normal([N, K[i - 1]]),
                        name='scale_z' + str(i)
                    )
                    if M != -1:
                        qWs['qW' + str(i - 1)] = TransformedDistribution(
                            distribution=Normal(
                                qW_variables['loc' + str(i - 1)],
                                tf.maximum(
                                    tf.nn.softplus(
                                        qW_variables['scale' + str(i - 1)]
                                    ),
                                    min_scale
                                )
                            ),
                            bijector=tf.contrib.distributions.bijectors.Exp()
                        )
                        qzs['qz' + str(i)] = TransformedDistribution(
                            distribution=Normal(
                                tf.gather(
                                    qz_variables['loc' + str(i)],
                                    idx_ph
                                ),
                                tf.maximum(
                                    tf.nn.softplus(
                                        tf.gather(
                                            qz_variables['scale' + str(i)],
                                            idx_ph
                                        )
                                    ),
                                    min_scale
                                )
                            ),
                            bijector=tf.contrib.distributions.bijectors.Exp()
                        )
                    else:
                        qWs['qW' + str(i - 1)] = TransformedDistribution(
                            distribution=Normal(
                                qW_variables['loc' + str(i - 1)],
                                tf.maximum(
                                    tf.nn.softplus(
                                        qW_variables['scale' + str(i - 1)]
                                    ),
                                    min_scale
                                )
                            ),
                            bijector=tf.contrib.distributions.bijectors.Exp()
                        )
                        qzs['qz' + str(i)] = TransformedDistribution(
                            distribution=Normal(
                                qz_variables['loc' + str(i)],
                                tf.maximum(
                                    tf.nn.softplus(
                                        qz_variables['scale' + str(i)]
                                    ),
                                    min_scale
                                )
                            ),
                            bijector=tf.contrib.distributions.bijectors.Exp()
                        )
                else:
                    qW_variables['shape' + str(i - 1)] = tf.Variable(
                        0.5 + 0.1 * tf.random_normal(
                            Ws['W' + str(i - 1)].shape
                        )
                    )
                    qW_variables['scale' + str(i - 1)] = tf.Variable(
                        0.1 * tf.random_normal(Ws['W' + str(i - 1)].shape)
                    )
                    qz_variables['shape' + str(i)] = tf.Variable(
                        0.5 + 0.1 * tf.random_normal([N, K[i - 1]])
                    )
                    qz_variables['scale' + str(i)] = tf.Variable(
                        0.1 * tf.random_normal([N, K[i - 1]])
                    )
                    if M != -1:
                        qWs['qW' + str(i - 1)] = Gamma(
                            tf.maximum(
                                tf.nn.softplus(
                                    qW_variables['shape' + str(i - 1)]
                                ),
                                min_shape
                            ),
                            tf.maximum(
                                1.0 / tf.nn.softplus(
                                    qW_variables['scale' + str(i - 1)]
                                ),
                                1.0 / min_scale
                            )
                        )
                        qzs['qz' + str(i)] = Gamma(
                            tf.maximum(
                                tf.nn.softplus(
                                    tf.gather(
                                        qz_variables['shape' + str(i)],
                                        idx_ph
                                    )
                                ),
                                min_shape
                            ),
                            tf.maximum(
                                1.0 / tf.nn.softplus(
                                    tf.gather(
                                        qz_variables['scale' + str(i)],
                                        idx_ph
                                    )
                                ),
                                1.0 / min_scale
                            )
                        )
                    else:
                        qWs['qW' + str(i - 1)] = Gamma(
                            tf.maximum(
                                tf.nn.softplus(
                                    qW_variables['shape' + str(i - 1)]
                                ),
                                min_shape
                            ),
                            tf.maximum(
                                1.0 / tf.nn.softplus(
                                    qW_variables['scale' + str(i - 1)]
                                ),
                                1.0 / min_scale
                            )
                        )
                        qzs['qz' + str(i)] = Gamma(
                            tf.maximum(
                                tf.nn.softplus(
                                    qz_variables['shape' + str(i)]
                                ),
                                min_shape
                            ),
                            tf.maximum(
                                1.0 / tf.nn.softplus(
                                    qz_variables['scale' + str(i)]
                                ),
                                1.0 / min_scale)
                        )

        if M != -1:
            x_ph = tf.placeholder(tf.float32, [M, D])

        inference_map = {}
        inference_data = {}
        scale_map = {}

        inference_w_map = {}
        inference_w_data = {}
        inference_z_map = {}
        inference_z_data = {}

        if M != -1:
            if map_estimate:
                inference_w_data[x] = x_ph
                inference_z_data[x] = x_ph
            else:
                inference_data[x] = x_ph
        else:
            if map_estimate:
                inference_w_data[x] = x_train
                inference_z_data[x] = x_train
            else:
                inference_data[x] = x_train

        for i in range(len(K)):
            if not map_estimate:
                inference_map[Ws['W' + str(i)]] = qWs['qW' + str(i)]
                inference_map[zs['z' + str(i + 1)]] = qzs['qz' + str(i + 1)]
            else:
                inference_w_map[Ws['W' + str(i)]] = qWs['qW' + str(i)]
                inference_w_data[zs['z' + str(i + 1)]] = qzs['qz' + str(i + 1)]
                inference_z_map[zs['z' + str(i + 1)]] = qzs['qz' + str(i + 1)]
                inference_z_data[Ws['W' + str(i)]] = qWs['qW' + str(i)]

            if M == -1:
                scale_map['z' + str(i + 1)] = float(N) / N
            else:
                scale_map['z' + str(i + 1)] = float(N) / M
                scale_map[x] = float(N) / M

        if map_estimate:
            inference_w = ed.MAP(inference_w_map, data=inference_w_data)
            inference_z = ed.KLqp(inference_z_map, data=inference_z_data)
            optimizer_w = tf.train.AdamOptimizer(lr)
            optimizer_z = tf.train.AdamOptimizer(lr)
        else:
            inference_func = ed.KLqp(inference_map, data=inference_data)
            optimizer_func = tf.train.AdamOptimizer(lr)
        timestamp = datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S")
        logdir += timestamp + '_' + '_'.join([str(ks) for ks in K]) + \
            '_q_' + str(q) + '_lr_' + str(lr)

        if map_estimate:
            inference_w.initialize(
                scale=scale_map,
                optimizer=optimizer_w
            )
            inference_z.initialize(
                scale=scale_map,
                control_variates=cv,
                logdir=logdir,
                n_print=100,
                optimizer=optimizer_z
            )
        else:
            inference_func.initialize(
                scale=scale_map,
                control_variates=False,
                logdir=logdir,
                optimizer=optimizer_func
            )

        perplexity_var = tf.Variable(0.0)
        tf.summary.scalar("perplexity", perplexity_var, collections=['test'])
        write_op = tf.summary.merge_all(key='test')

        x_test_head_start = np.random.binomial(n=x_test, p=0.2)
        x_test_remaining = x_test - x_test_head_start
        N_test = len(x_test_head_start)

        Ws_test = {}
        Ws_test['W0'] = Gamma(0.1, 0.3, sample_shape=[K[0], D])
        for i in range(1, len(K)):
            Ws_test['W' + str(i)] = Gamma(
                0.1,
                0.3,
                sample_shape=[K[i], K[i - 1]]
            )

        zs_test = {}
        zs_test['z' + str(len(K))] = Gamma(
            0.1,
            0.1,
            sample_shape=[N_test, K[-1]]
        )
        for i in range(len(K) - 1, 0, -1):
            scale_denominator_test = tf.matmul(
                zs_test['z' + str(i + 1)],
                Ws_test['W' + str(i)]
            )
            if skip_connections & (i % 3 == 1):
                scale_denominator_test = scale_denominator_test + zs_test[
                    'z' + str(i + 2)
                ]
            if most_skip:
                if i % 3 == 1:
                    scale_denominator_test = scale_denominator_test + zs_test[
                        'z' + str(i + 2)
                    ] + zs_test[
                        'z' + str(i + 1)
                    ]
                if i % 3 == 2:
                    scale_denominator_test = scale_denominator_test + zs_test[
                        'z' + str(i + 1)
                    ]
            zs_test['z' + str(i)] = Gamma(
                shape,
                shape / scale_denominator_test
            )
        x_t = Poisson(tf.matmul(zs_test['z1'], Ws_test['W0']))

        qz_variables_test = {}
        qzs_test = {}
        for i in range(1, len(K) + 1):
            if q == 'lognormal':
                qz_variables_test['loc' + str(i)] = tf.Variable(
                    tf.random_normal([N_test, K[i - 1]])
                )
                qz_variables_test['scale' + str(i)] = tf.Variable(
                    0.1 * tf.random_normal([N_test, K[i - 1]])
                )
                qzs_test['qz' + str(i)] = TransformedDistribution(
                    distribution=Normal(
                        qz_variables_test['loc' + str(i)],
                        tf.maximum(
                            tf.nn.softplus(
                                qz_variables_test['scale' + str(i)]
                            ),
                            min_scale
                        )
                    ),
                    bijector=tf.contrib.distributions.bijectors.Exp()
                )
            else:
                qz_variables_test['shape' + str(i)] = tf.Variable(
                    0.5 + 0.1 * tf.random_normal([N_test, K[i - 1]])
                )
                qz_variables_test['scale' + str(i)] = tf.Variable(
                    0.1 * tf.random_normal([N_test, K[i - 1]])
                )
                qzs_test['qz' + str(i)] = Gamma(
                    tf.maximum(
                        tf.nn.softplus(qz_variables_test['shape' + str(i)]),
                        min_shape
                    ),
                    tf.maximum(
                        1.0 / tf.nn.softplus(
                            qz_variables_test['scale' + str(i)]
                        ),
                        1.0 / min_scale
                    )
                )

        inference_z_data_test = {}
        inference_z_map_test = {}
        inference_z_data_test[x_t] = x_test_head_start

        for i in range(len(K)):
            inference_z_map_test[zs_test['z' + str(i + 1)]] = qzs_test[
                'qz' + str(i + 1)
            ]
            inference_z_data_test[Ws_test['W' + str(i)]] = qWs['qW' + str(i)]
        inference_z_test = ed.KLqp(
            inference_z_map_test,
            data=inference_z_data_test
        )

        optimizer_z_test = tf.train.AdamOptimizer(lr)
        inference_z_test.initialize(n_print=100, optimizer=optimizer_z_test)
        qz_init = tf.variables_initializer(list(qz_variables_test.values()))

        sess = ed.get_session()
        tf.global_variables_initializer().run()

        print("Log Directory: ", logdir)
        losses = []
        test_perps = []
        for epoch in range(n_epoch):
            pbar = Progbar(n_iter_per_epoch)
            for t in range(1, n_iter_per_epoch + 1):
                pbar.update(t)
                if M != -1:
                    x_batch, idx_batch = next_batch(M)
                    if map_estimate:
                        info_dict = inference_z.update(
                            feed_dict={x_ph: x_batch, idx_ph: idx_batch}
                        )
                        inference_w.update(
                            feed_dict={x_ph: x_batch, idx_ph: idx_batch}
                        )
                    else:
                        info_dict = inference_func.update(
                            feed_dict={x_ph: x_batch, idx_ph: idx_batch}
                        )
                else:
                    if map_estimate:
                        info_dict = inference_z.update()
                        inference_w.update()
                    else:
                        info_dict = inference_func.update()
                losses.append(info_dict['loss'])

            if map_estimate:
                qW0_vals = sess.run(qWs['qW0'])
            else:
                if q == 'gamma':
                    qW0_vals = sess.run(qWs['qW0'].mean())
                else:
                    qW0_vals = sess.run(tf.exp(
                        qWs['qW0'].distribution.mean() +
                        qWs['qW0'].distribution.variance() / 2
                    ))

            pbar_test = Progbar(n_test_epoch * n_test_iter)
            for epoch_test in range(n_test_epoch):

                for t in range(1, n_test_iter + 1):
                    pbar_test.update(epoch_test * n_test_iter + t)
                    inference_z_test.update()

                if q == 'gamma':
                    z1_mean = sess.run(qzs_test['qz1'].mean())
                else:
                    z1_mean = sess.run(tf.exp(
                        qzs_test['qz1'].distribution.mean() +
                        qzs_test['qz1'].distribution.variance() / 2
                    ))
                w0_mean = qW0_vals

                doc_rate = np.matmul(z1_mean, w0_mean)
                doc_rate_normalized = doc_rate / np.sum(
                    doc_rate,
                    axis=1
                ).reshape(len(doc_rate), 1)
                test_perp = np.exp(
                    -1 * np.sum(
                        np.log(doc_rate_normalized) *
                        x_test_remaining
                    ) / np.sum(x_test_remaining)
                )
                test_perps.append(test_perp)

                summary = sess.run(write_op, {perplexity_var: test_perp})
                if not map_estimate:
                    inference_func.train_writer.add_summary(
                        summary,
                        epoch * n_test_epoch + epoch_test
                    )
                else:
                    inference_z.train_writer.add_summary(
                        summary,
                        epoch * n_test_epoch + epoch_test
                    )

            qz_init.run()

        savedir = savedir + '/'
        if not os.path.exists(os.path.dirname(savedir)):
            os.makedirs(os.path.dirname(savedir))
        final_zs = {}
        final_Ws = {}
        for i in range(len(K)):
            if map_estimate:
                final_Ws[i] = sess.run(qWs['qW' + str(i)])
                if q == 'gamma':
                    final_zs[i + 1] = sess.run(qzs['qz' + str(i + 1)].mean())
                else:
                    final_zs[i + 1] = sess.run(tf.exp(
                        qzs['qz' + str(i + 1)].distribution.mean() +
                        qzs['qz' + str(i + 1)].distribution.variance() / 2
                    ))
            else:
                if q == 'gamma':
                    final_zs[i + 1] = sess.run(qzs['qz' + str(i + 1)].mean())
                    final_Ws[i] = sess.run(qWs['qW' + str(i)].mean())
                else:
                    final_zs[i + 1] = sess.run(tf.exp(
                        qzs['qz' + str(i + 1)].distribution.mean() +
                        qzs['qz' + str(i + 1)].distribution.variance() / 2
                    ))
                    final_Ws[i] = sess.run(tf.exp(
                        qWs['qW' + str(i)].distribution.mean() +
                        qWs['qW' + str(i)].distribution.variance() / 2
                    ))
            np.save(savedir + '/z' + str(i + 1), final_zs[i + 1])
            np.save(savedir + '/W' + str(i), final_Ws[i])
        np.save(savedir + '/test_perplexities', np.array(test_perps))
        np.save(savedir + '/losses', np.array(losses))

        tf.reset_default_graph()

        return np.array(test_perps), np.array(losses), final_zs, final_Ws
