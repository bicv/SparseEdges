"""
EdgeFactory

See http://pythonhosted.org/SparseEdges

"""
__author__ = "Laurent Perrinet INT - CNRS"
__licence__ = 'GPLv2'
import numpy as np
import os
# import socket
# PID, HOST = os.getpid(), socket.gethostname()
# TAG = 'host-' + HOST + '_pid-' + str(PID)
# -------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
import sys, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

from SparseEdges import SparseEdges, KL, TV

class EdgeFactory(SparseEdges):
    """
    EdgeFactory

    A class which classifies images based on histograms of their statistics.

    We use the ``SVM`` classifier from sklearn.

    The pipeline is to

    * extract edges from all images,
    * create the representation (histogram) for each class,
    * fit the data,
    * classify and return the f1-score.

    """
    def __init__(self, pe):
        """
        Initializes the SparseEdges class

        """
        SparseEdges.__init__(self, pe)
        self.init()
        self.init_logging(name='EdgeFactory')

    def init(self):
        SparseEdges.init(self)

    def get_labels(self, edges, filename, croparea, database_labels=None, pos_noise=None, seed=None):
        # see "2018-07-09 loading ground truth - associating labels to edges" to tune pos_noise
        if database_labels is None: database_labels = self.pe.database_labels
        if pos_noise is None: pos_noise = self.pe.pos_noise
        if seed is None: seed = self.pe.seed
        np.random.seed(seed)

        labels = np.load(os.path.join(database_labels, filename.replace('.png', '.npy')))
        labels = labels[croparea[0]:croparea[1], croparea[2]:croparea[3]]
        X, Y, sf_0 = edges[0, :], edges[1, :],  edges[3, :]
        if pos_noise>0:
            X += pos_noise * np.random.randn(edges.shape[1]) / sf_0
            Y += pos_noise * np.random.randn(edges.shape[1]) / sf_0
        X = [int(np.max((0, np.min((X_, self.pe.N_X-1))))) for X_ in X]
        Y = [int(np.max((0, np.min((Y_, self.pe.N_Y-1))))) for Y_ in Y]
        y = labels[X, Y].astype(np.int)
        return labels, y

    def svm(self, exp, opt_notSVM='', opt_SVM='', databases=['serre07_distractors', 'serre07_targets'],
            edgeslists=[None, None], N_edges=None, database_labels=None, group_labels=None,
            feature='full', kernel='precomputed', KL_type='JSD', noise=0.):
        """
        outline:

         - gathering all data:
          * first the raw complete data,
          * then the X, y vectors;
        -the final result is the matname_score file containg the
        classification results


        by convention, if we use a database_labels string (path to folder),
        then we perform an edge-by-edge classification

        """
        import time
        time.sleep(.1*np.random.rand())
        # DEFINING FILENAMES
        # put here thing that do change the histogram
        # opt_notSVM = opt_notSVM # (this should be normally passed to exp, as in 'classifier_noise')
        # and here things that do not change the (individual) histograms but
        # rather the SVM classification:
        if (self.pe.svm_log): opt_SVM += '_log'
#         if not(self.pe.svm_log): opt_SVM += 'nolog_'
        if self.pe.svm_norm: opt_SVM += '_norm'
        name_databases = ''
        for database in databases: name_databases += database + '_'
        # print('name_database', name_database)  #DEBUG

        txtname = os.path.join(self.pe.figpath, exp + '_SVM_' + name_databases + feature + opt_notSVM + opt_SVM +'.txt')
        matname_score = txtname.replace(self.pe.figpath, self.pe.matpath).replace('.txt', '.npy')


        # DEFINING FEATURES TO USE
        if feature == 'first_chevron':
            features = ['first', 'chevron']
        elif feature == 'first_full':
            features = ['first', 'full']
        else:
            features = [feature]
        ###############################################################################
        # Process all images to extract edges and plot relevant histograms
        n_databases = len(databases)
        mode = 'full' if n_databases>1 else 'edge'
        # print('mode', mode)  #DEBUG
        for i_database, (name_database, edgeslist) in enumerate(zip(databases, edgeslists)):
            if edgeslist is None:
                imagelist, edgeslist, MSE = self.process(exp, note=opt_notSVM, name_database=name_database, noise=noise)
            else:
                imagelist = 'ok'

        # print('matname_score', matname_score)  #DEBUG
        if os.path.isfile(matname_score):
            fone_score = np.load(matname_score)
            self.log.warn("=> Accuracy = %0.2f +/- %0.2f in %s ", fone_score.mean(), fone_score.std(), txtname)
            return fone_score

        if os.path.isfile(matname_score + '_lock'):
            self.log.info(' >> Locked SVM : %s ', matname_score + '_lock')
            return None
        else:
            open(matname_score + '_lock', 'w').close()
            for feature_ in features:
                ###############################################################################
                # Download the data, if not already on disk and saves it as numpy arrays
                n_databases = len(databases)

                for i_database, (name_database, edgeslist) in enumerate(zip(databases, edgeslists)):
                    matname_hist = os.path.join(self.pe.matpath, exp + '_SVM-hist_' + name_database + '_' + feature_ + opt_notSVM + '.npy')

                    if not(os.path.isfile(matname_hist)):
                        self.log.info(' >> There is no histogram, computing %s ', matname_hist)
                        if os.path.isfile(matname_hist + '_lock'):
                            self.log.info(' XX The process computing the histogram in %s is locked by %s_lock', name_database, matname_hist)
                        else:
                            open(matname_hist + '_lock', 'w').close()
                            if edgeslist is None:
                                imagelist, edgeslist, MSE = self.process(exp, note=opt_notSVM, name_database=name_database, noise=noise)
                            else:
                                imagelist = 'ok'
                            try:
                                t0 = time.time()
                                hists = []
                                # TODO : compute histograms with less edges?
                                N_edges_hist = edgeslist.shape[1]

                                for i_image in range(edgeslist.shape[2]):
                                    if feature_ == 'full':
                                        # using the full histogram
                                        v_hist = self.cooccurence_hist(edgeslist[:, :N_edges_hist, i_image], mode=mode)
                                    elif feature_ == 'full_nochevron':
                                        #  or just the chevron map
                                        v_hist = self.cooccurence_hist(edgeslist[:, :N_edges_hist, i_image], mode=mode)
                                        # marginalize over theta and psi
                                        v_hist = v_hist.sum(axis=(1, 2))
                                    elif feature_ == 'chevron':
                                        #  or just the chevron map
                                        v_hist = self.cooccurence_hist(edgeslist[:, :N_edges_hist, i_image], mode=mode)
                                        # marginalize over distances and scales
                                        v_hist = v_hist.sum(axis=(0, 3))
                                    elif feature_ == 'first':
                                        # control with first-order
                                        v_hist, v_theta_edges_ = self.histedges_theta(edgeslist[:, :N_edges_hist, i_image], display=False, mode=mode)
                                    elif feature_ == 'first_rot':
                                        edgeslist[2, :, i_image] += np.random.rand() * np.pi
                                        # control with first-order
                                        v_hist, v_theta_edges_ = self.histedges_theta(edgeslist[:, :N_edges_hist, i_image], display=False, mode=mode)
                                    else:
                                        self.log.error('problem here, you asked for a non-existant feature', feature_)
                                        break

                                    # normalize histogram
                                    if mode=='full':
                                        v_hist /= v_hist.sum()
                                    else:
                                        # take only the first edges
                                        # but still, the histogram is computed on all edges
                                        if N_edges is None: N_edges = edgeslist.shape[1]
                                        v_hist = v_hist[..., :N_edges]
                                        for i_edge in range(v_hist.shape[-1]):
                                            if v_hist[..., i_edge].sum() ==0 : print('dooh! v_hist is null')
                                            v_hist[..., i_edge] /= v_hist[..., i_edge].sum()

                                    # append for each image
                                    hists.append(v_hist)

                                hists = np.array(hists)
                                np.save(matname_hist, hists)
                                self.log.info("Histogram done in %0.3fs", (time.time() - t0))
                            except Exception as e:
                                self.log.error(' XX The process computing edges in %s is locked ', name_database)
                                self.log.error(' Raised Exception %s  ', e)
                            try:
                                os.remove(matname_hist + '_lock')
                            except Exception:
                                self.log.error(' xxx when trying to remove it, I found no lock file named %s_lock', matname_hist)


            # gather data
            # TODO faster reshaping
            locked = False
            X_, y_ = {}, []
            for feature_ in features:
                X_[feature_] = []
                for i_database, name_database in enumerate(databases):
                    matname_hist = os.path.join(self.pe.matpath, exp + '_SVM-hist_' + name_database + '_' + feature_ + opt_notSVM + '.npy')
                    try:
                        hists = np.load(matname_hist)
                        # print('hists ratio negative', (hists<0).sum() / hists.size) #DEBUG

                        for i_image in range(hists.shape[0]):
                            if mode=='full':
                                X_[feature_].append(hists[i_image, ...].ravel())
                            else:
                                X__ = []
                                for i_edge in range(hists.shape[-1]):
                                    X__.append(hists[i_image, ..., i_edge].ravel())
                                X_[feature_].append(X__)
                                # print( 'np.array(X__).shape', np.array(X__).shape, 'np.array(X_[feature_]).shape', np.array(X_[feature_]).shape)  #DEBUG

                    except Exception as e:
                        self.log.warn(' >> Missing histogram, skipping SVM : %s ', e)
                        locked = True
                        return None

            # appending all data for all images
            for i_database, (name_database, edgeslist) in enumerate(zip(databases, edgeslists)):
                imagelist, edgeslist, MSE = self.process(exp, note=opt_notSVM, name_database=name_database, noise=noise)
                try:
                    if mode=='full':
                        for i_image, (filename, croparea) in enumerate(imagelist):
                            y_.append(i_database)
                    else:
                        if N_edges is None:
                            N_edges = edgeslist.shape[1]
                        # print('N_edges', N_edges)  #DEBUG
                        for i_image, (filename, croparea) in enumerate(imagelist):
                            labels, y__ = self.get_labels(edgeslist[:, :N_edges, i_image], filename, croparea, database_labels=database_labels)
                            if not group_labels is None:
                                y__ = [group_labels[label] for label in y__]
                            y_.append(y__)
                except Exception as e:
                    self.log.warn(' >> Failed the labelling, skipping SVM : %s ', e)
                    locked = True
                    return None

            # converting to numpy
            X = {}
            for feature_ in features:
                X[feature_] = np.array(X_[feature_])
                # print('feature_', feature_, 'X[feature_].shape', X[feature_].shape)  #DEBUG
            y = np.array(y_)
            # print('y.shape', y.shape)  #DEBUG
            # print('y.shape', y.shape, 'y', y)  #DEBUG


            # do the classification
            fone_score = np.zeros(self.pe.N_svm_cv)
            t0_cv = time.time()
            for i_cv in range(self.pe.N_svm_cv):
                # print('i_cv', i_cv, 'self.pe.N_svm_cv', self.pe.N_svm_cv, 'fone_score', fone_score)  #DEBUG

                ###############################################################################
                # 1- Split into a training set and a test set using a ShuffleSplit + doing that in parallel for the differrent features to test
                from sklearn.model_selection import ShuffleSplit
                rs = ShuffleSplit(n_splits=1, test_size=self.pe.svm_test_size, random_state=self.pe.seed + i_cv)
                # split into a training and testing set
                for index_train, index_test in rs.split(y): pass
                # print('index_train', index_train, 'index_test', index_test)  #DEBUG
                X_train, X_test = {}, {}
                if mode=='full':
                    for feature_ in features:
                        X_train[feature_], X_test[feature_] = X[feature_][index_train, :], X[feature_][index_test, :]
                    y_train, y_test =  y[index_train].copy(), y[index_test].copy()
                else:
                    for feature_ in features:
                        X_train[feature_], X_test[feature_] = X[feature_][index_train, :, :].copy(), X[feature_][index_test, :, :].copy()
                    y_train, y_test =  y[index_train, :].copy(), y[index_test, :].copy()
                n_train, n_test = y_train.shape[0], y_test.shape[0]
                # print('n_train', n_train, 'n_test', n_test)  #DEBUG
                # is_target.append(y_test)
                # tested_indices.append(index_test)

                # 2- normalization TODO: check in mode edge if that normalization is fine
                if self.pe.svm_log and (kernel == 'rbf'):
                    # trying out if some log-likelihood like representation is better for classification (makes sense when thinking that animals would have some profile modulating all probabilities)
                    eps = 1.e-16
                    for feature_ in features:
                        m_hist_1 = X_train[feature_][y_train==1, ...].mean(axis=0) # average histogram for distractors x on the training set
                        for i_image in range(X_train[feature_].shape[0]): X_train[feature_][i_image, ...] = np.log(X_train[feature_][i_image, ...] + eps)-np.log(m_hist_1 + eps)
                        for i_image in range(X_test[feature_].shape[0]): X_test[feature_][i_image, ...] = np.log(X_test[feature_][i_image, ...] + eps)-np.log(m_hist_1 + eps)
                if self.pe.svm_norm:
                    if (kernel == 'rbf'):
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        for feature_ in features:
                            X_train[feature_] = scaler.fit_transform(X_train[feature_])
                            scaler.fit(X[feature_])
                            X_test[feature_] = scaler.transform(X_test[feature_])
                    else: # applying a "prior" (the probability represents the probability /knowing/ it belongs to the reference set containging all categories)
                        eps = 1.e-16
                        for feature_ in features:
                            m_hist_1 = X_train[feature_][y_train==1, ...].mean(axis=0) # average histogram for distractors x on the training set
                            for i_image in range(n_train):
                                X_train[feature_][i_image, ...] = X_train[feature_][i_image, ...]/(m_hist_1 + eps)
                                X_train[feature_][i_image, ...] /= X_train[feature_][i_image, ...].sum()
                            for i_image in range(n_test):
                                X_test[feature_][i_image, ...] = X_test[feature_][i_image, ...]/(m_hist_1 + eps)
                                X_test[feature_][i_image, ...] /= X_test[feature_][i_image, ...].sum()

                try:
                    # sanity check with a dummy classifier:
                    from sklearn.dummy import DummyClassifier
                    from sklearn import metrics
                    dc = DummyClassifier(strategy='most_frequent', random_state=self.pe.seed+i_cv)
                    X_train_, X_test_ = np.zeros((n_train, 0)), np.zeros((n_test, 0))
                    for feature_ in features:
                        X_test_ = np.hstack((X_test_, X_test[feature_]))
                        X_train_ = np.hstack((X_train_, X_train[feature_]))
                    if y_train.size > 0:
                        dc = dc.fit(X_train_, y_train)
                        self.log.warn("Sanity check with a dummy classifier:")
                        self.log.warn("score = %f ", dc.score(X_test_, y_test))#, scoring=metrics.f1_score))
                except Exception as e:
                    self.log.error("Failed doing the dummy classifier : %s ", e)
                ###############################################################################
                ###############################################################################
                # 3- preparing the gram matrix
                if y_train.size == 0:
                    self.log.error("preparing the gram matrix but y_train.size == 0 ")
                    break

                if not(kernel == 'rbf'):
                    # use KL distance as my kernel
                    kernel = 'precomputed'
                    def distance(x, y, KL_type=KL_type):
                        if KL_type=='sKL': return (KL(x, y) + KL(y, x))#symmetric KL
                        elif KL_type=='JSD': return (KL(x, (x+y)/2.) + KL(y, (x+y)/2.))/2.#Jensen-Shannon divergence
                        elif KL_type=='TV': return TV(x, y) # Total variation
                        else: return KL(x, y)

                    def my_kernel(d, KL_m, use_log, KL_0):
                        if use_log:
                            return d/KL_m
                        else:
                            return np.exp(-d/KL_m/KL_0)

                    if mode=='full':
                        n_train = y_train.shape[0]
                        n_test = y_test.shape[0]
                    else:
                        n_train = y_train.shape[0]*y_train.shape[1]
                        n_test = y_test.shape[0]*y_test.shape[1]

                    gram_train = np.zeros((n_train, n_train))
                    gram_test = np.zeros((n_train, n_test))
                    for feature_ in features:
                        # print('feature_', feature_, 'X_train[feature_].shape', X_train[feature_].shape)  #DEBUG
                        # print('feature_', feature_, 'X_test[feature_].shape', X_test[feature_].shape)  #DEBUG

                        # compute the average KL
                        KL_0 = 0
                        if mode=='full':
                            for i_ in range(X_train[feature_].shape[0]):
                                for j_ in range(X_train[feature_].shape[0]):
                                    KL_0 += distance(X_train[feature_][i_, :], X_train[feature_][j_, :], KL_type=KL_type)
                        else:
                            X_train_ = X_train[feature_].reshape((n_train, X_train[feature_].shape[-1]))
                            # print('feature_', feature_, 'X_train_.shape', X_train_.shape)  #DEBUG
                            KL_train = distance(X_train_, X_train_, KL_type=KL_type)
                            # print('feature_', feature_, 'KL_train.shape', KL_train.shape)  #DEBUG
                            # print('KL_train ratio negative', (KL_train<0).sum() / KL_train.size) #DEBUG
                            KL_0 = KL_train.sum()
                            # X_train[feature_].shape[0]
                            # for i_image_ in range(X_train[feature_].shape[0]):
                            #     print(' KL_0 i_image_', i_image_, '/X_train[feature_].shape[0]=', X_train[feature_].shape[0])  #DEBUG
                            #     for i_edge_ in range(X_train[feature_].shape[1]):
                            #         print(' KL_0 i_edge_', i_edge_, '/X_train[feature_].shape[1]=', X_train[feature_].shape[1])  #DEBUG
                            #         for j_image_ in range(X_train[feature_].shape[0]):
                            #             for j_edge_ in range(X_train[feature_].shape[1]):
                            #                 KL_0 += distance(X_train[feature_][i_image_, i_edge_, :], X_train[feature_][j_image_, j_edge_, :], KL_type=KL_type)
                        KL_0 /= n_train**2
                        self.log.info('KL_0 = %f ', KL_0)
                        self.log.info('for feature_ = %s ', feature_)
                        # print('KL_0 = ', KL_0, 'for feature_ = ', feature_)  #DEBUG

                        if mode=='full':
                            # TODO : vectorize
                            for i_ in range(n_train):
                                for j_ in range(n_train):
                                    d = distance(X_train[feature_][i_, :], X_train[feature_][j_, :])
                                    gram_train[i_, j_] += my_kernel(d, KL_m=self.pe.svm_KL_m, use_log=self.pe.svm_log, KL_0=KL_0)
                            # TODO: check the fact that we compute the crossed gram maytrix with train to test distance
                            for i_ in range(n_train):
                                for j_ in range(n_test):
                                    d = distance(X_train[feature_][i_, :], X_test[feature_][j_, :])
                                    gram_test[i_, j_] += my_kernel(d, KL_m=self.pe.svm_KL_m, use_log=self.pe.svm_log, KL_0=KL_0)
                        else:
                            # cf 2018-07-24 associating labels to edges-SVM
                            # for i_image_ in range(X_train[feature_].shape[0]):
                            #     print('gram_train i_image_', i_image_, 'X_train[feature_].shape[0]', X_train[feature_].shape[0])  #DEBUG
                            #     for i_edge_ in range(X_train[feature_].shape[1]):
                            #         for j_image_ in range(X_train[feature_].shape[0]):
                            #             for j_edge_ in range(X_train[feature_].shape[1]):
                            #                 i__ = i_image_*X_train[feature_].shape[1]+i_edge_
                            #                 j__ = j_image_*X_train[feature_].shape[1]+j_edge_
                            #                 gram_train[i__, j__] += my_kernel(X_train[feature_][i_image_, i_edge_, :], X_train[feature_][j_image_, j_edge_, :], KL_m=self.pe.svm_KL_m, use_log=self.pe.svm_log, KL_0=KL_0)
                            gram_train =  my_kernel(KL_train, KL_m=self.pe.svm_KL_m, use_log=self.pe.svm_log, KL_0=KL_0)

                            # for i_image_ in range(X_train[feature_].shape[0]):
                            #     print('gram_test i_image_', i_image_, 'X_train[feature_].shape[0]', X_train[feature_].shape[0])  #DEBUG
                            #     for i_edge_ in range(X_train[feature_].shape[1]):
                            #         for j_image_ in range(X_test[feature_].shape[0]):
                            #             for j_edge_ in range(X_test[feature_].shape[1]):
                            #                 i__ = i_image_*X_train[feature_].shape[1]+i_edge_
                            #                 j__ = j_image_*X_test[feature_].shape[1]+j_edge_
                            #                 gram_test[i__, j__] += my_kernel(X_train[feature_][i_image_, i_edge_, :], X_test[feature_][j_image_, j_edge_, :], KL_m=self.pe.svm_KL_m, use_log=self.pe.svm_log, KL_0=KL_0)
                            X_test_ = X_test[feature_].reshape((n_test, X_test[feature_].shape[-1]))
                            KL_test = distance(X_train_, X_test_, KL_type=KL_type)
                            gram_test =  my_kernel(KL_test, KL_m=self.pe.svm_KL_m, use_log=self.pe.svm_log, KL_0=KL_0)

                    # print('number of nan in gram_train=', np.isnan(gram_train).sum())  #DEBUG
                    # print('number of nan in gram_test=', np.isnan(gram_test).sum())  #DEBUG
                ###############################################################################
                # 4- Train a SVM classification model
                from sklearn.model_selection import GridSearchCV
                # see http://scikit-learn.org/stable/modules/grid_search.html
                from sklearn.svm import SVC
                # from sklearn import model_selection
                self.log.info("Fitting the classifier to the training set %s - %s - %s ",  databases, exp, feature)
                t0 = time.time()
                if kernel == 'precomputed':
                    C_range = np.logspace(self.pe.C_range_begin,self.pe.C_range_end, self.pe.N_svm_grid**2, base=2.)
                    param_grid = {'C': C_range }
                else:
                    C_range = np.logspace(self.pe.C_range_begin, self.pe.C_range_end, self.pe.N_svm_grid, base=2.)
                    gamma_range = np.logspace(self.pe.gamma_range_begin,self.pe.gamma_range_end, self.pe.N_svm_grid, base=2.)
                    param_grid = {'C': C_range, 'gamma': gamma_range }
                grid = GridSearchCV(SVC(verbose=False,
                                        kernel=kernel,
                                        tol=self.pe.svm_tol,
    #                                             probability=True,
                                        class_weight='balanced',
                                        max_iter = self.pe.svm_max_iter,
                                        ),
                                    param_grid,
                                    verbose=1,
                                    scoring='f1_weighted',
                                    cv=self.pe.N_svm_cv,
                                    n_jobs=self.pe.svm_n_jobs, # http://scikit-learn.org/0.13/modules/generated/sklearn.grid_search.GridSearchCV.html
                                    #pre_dispatch=2*self.pe.svm_n_jobs,
                                    )
                if kernel == 'precomputed':
                    print ('gram_train.shape=', gram_train.shape, 'y_train.shape=', y_train.shape)   #DEBUG
                    grid.fit(gram_train, y_train.ravel())
                else:
                    X_train_ = np.zeros((n_train, 0))
                    for feature_ in features:
                        X_train_ = np.hstack((X_train_, X_train[feature_]))
                    grid.fit(X_train_, y_train.ravel())

                self.log.info("Fitting the classifier done in %0.3fs", (time.time() - t0))
                if self.log.level <= 10:
                    t0 = time.time()
                    self.log.info("Predicting the category names on the learning set")
                    if kernel == 'precomputed':
                        y_pred = grid.predict(gram_train)
                    else:
                        y_pred = grid.predict(X_train_.reshape((n_train, X_train[feature_].shape[-1])))
                    from sklearn.metrics import classification_report
    #                             print y_train, y_pred
    # TODO                         self.log.info(classification_report(y_train, y_pred))
                    self.log.info("Prediction done in %0.3fs" % (time.time() - t0))
                    if self.log.level<=10:
                        self.log.info("For %s the best estimator found by grid search is:", exp + opt_SVM)
    # TODO                         self.log.info(grid.best_estimator_)
                        print(grid.best_estimator_)
#                         print "Grid scores on development set:"
#                         for params, mean_score, scores in grid.grid_scores_:
#                             print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)

                if i_cv==0:  # TODO: draw for all CV
                    try:
                        figname = txtname.replace('.txt', '_grid.' + self.pe.formats[0])
                        # plot the scores of the grid
                        # grid_scores_ contains parameter settings and scores
                        score_dict = grid.grid_scores_
                        scores_mean, scores_std =[],  []
                        for params, mean_score, scores in score_dict:
                            scores_mean.append(scores.mean())
                            scores_std.append(scores.std()/2)

                        # draw heatmap of accuracy as a function of gamma and C
                        fig = plt.figure(figsize=(8, 6))
                        if kernel == 'precomputed':
                            ax = fig.add_subplot(1, 1, 1)
                            ax.errorbar(C_range, np.array(scores_mean), yerr=np.array(scores_std))
                            ax.set_xscale('log')
                            plt.xlabel('C')
                            plt.ylabel('f1_score')
                            plt.axis('tight')
                        else:
                            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
                            scores = np.array(scores_mean).reshape((gamma_range.shape[0], C_range.shape[0]))
                            plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
                            plt.xlabel('gamma')
                            plt.ylabel('C')
                            plt.colorbar()
                            N_step = np.floor(len(gamma_range) / 5)
                            plt.xticks(np.arange(0, len(gamma_range), N_step), ['2^%.2f' % np.log2(k) for k in gamma_range[::N_step]], rotation=45)
                            N_step = np.floor(len(C_range) / 5)
                            plt.yticks(np.arange(0, len(C_range), N_step), ['2^%.2f' % np.log2(k) for k in C_range[::N_step]])
                        self.savefig(fig, figname)
                    except Exception as e:
                        self.log.error('could not draw grid score : %s ', e)

                ###############################################################################
                # 5- Quantitative evaluation of the model quality on the test set
                t0 = time.time()
                self.log.info("Predicting the category names on the testing set")
                if kernel == 'precomputed':
                    y_pred = grid.predict(gram_test.T)
                else:
                    X_test_ = np.zeros((n_test, 0))
                    for feature_ in features:
                        X_test_ = np.hstack((X_test_, X_test[feature_]))
                    y_pred = grid.predict(X_test_)
                # predicted_target.append(y_pred)
                if self.log.level<=10:
                    from sklearn.metrics import classification_report
                    print('classification_report on test \n', classification_report(y_test.ravel(), y_pred))
                    from sklearn.metrics import confusion_matrix
                    print('confusion_matrix on test \n', confusion_matrix(y_test.ravel(), y_pred))
                    print('fone_score on test', metrics.f1_score(y_test.ravel(), y_pred))
                # see https://en.wikipedia.org/wiki/F1_score
                try:
                    fone_score[i_cv] = np.array(metrics.f1_score(y_test.ravel(), y_pred, average=None)).mean()#'weighted')labels=[0, 1],
                except Exception:
                    self.log.error(' something bad happened for the fone score ')
                results = "=> Accuracy @ %d = %0.2f" % (i_cv+1, fone_score[i_cv])
                results += " in " + txtname
                print(results)
                results += "\n"
                if i_cv > 0:
                    with open(txtname, 'a') as f: f.write(results)
                else:
                    with open(txtname, 'w') as f: f.write(results)
                self.log.info("Prediction on the testing set done in %0.3fs", (time.time() - t0))

#                 if True: # edgeslist is None: # ???
#                     self.log.info(">> compiling results ")
#                     t0_matname_score_dic = time.time()
#                     # tested_indices is the index of the image that is tested
#                     # is_target is 1 if it is a target
#                     # predicted_target is the response of the categorizer
#                     imagelists = [self.get_imagelist(exp, name_database=databases[0]),
#                                   self.get_imagelist(exp, name_database=databases[1])]
#                     N_image = len(imagelists[0])
#                     matname_score_dic = txtname.replace(self.pe.figpath, self.pe.matpath).replace('.txt', '.pickle')
#                     try:
#                         with open(matname_score_dic, "wb" ) as f:
#                             results = pickle.load(f)
#                     except Exception:
#                         results = {}
#                         # setting up dictionary counting for each file how many times (integer) it is tested in total, how many times it is a target
#                         for i_database in range(2):
#                             for filename_, croparea_ in imagelists[i_database]:
#                                 results[filename_] = [0, 0]
#
# #                     for vec in [tested_indices, is_target, predicted_target]: print len(vec)
#                     for i_image, in_cat, pred_cat in zip(np.array(tested_indices).ravel(), np.array(is_target).ravel(), np.array(predicted_target).ravel()):
# #                         print i_image, in_cat, pred_cat
#                         filename_, croparea_ = imagelists[in_cat][i_image - in_cat*N_image]
#                         results[filename_][0] +=  1 # how many times in total
#                         results[filename_][1] +=  1*(pred_cat==1) # how many times it is a target
#                     with open(matname_score_dic, "wb" ) as f:
#                         pickle.dump(results, f)
#                     self.log.info("Computing matname_score_dic done in %0.3fs", (time.time() - t0_matname_score_dic))
                t_cv = time.time()
                self.log.warn('Cross-validation in %s (%d/%d) - elaspsed = %0.1f s - ETA = %0.1f s ' % (matname_score, i_cv+1, self.pe.N_svm_cv, t_cv-t0_cv, (self.pe.N_svm_cv-i_cv-1)*(t_cv-t0_cv)/(i_cv+1) ) )

        try:
            np.save(matname_score, fone_score)
        except IOError as e:
            self.log.error('error %s while making %s ', e, matname_score)#, fone_score

        try:
            os.remove(matname_score + '_lock')
        except Exception:
            self.log.error(' no matname_score lock file named %s_lock ', matname_score)

        return fone_score

    def compare(self, exp, databases=['serre07_distractors', 'serre07_targets'],
                noise=0., geometric=False, rho_quant=128, do_scale=True):
        """
        Here, we compare 2 sets of images thanks to their respective histograms
        of edge co-occurences using a 2-means classification algorithm


        """
        v_hist, edgeslist_db = [], []
        #############################
        locked = False # check whether edge extraction is finished
        self.log.info(' > comparing second-order statistics for experiment %s', exp)
        for name_database in databases:
            matname = os.path.join(self.pe.matpath, exp + '_' + name_database)
            self.log.info(' >> getting edges for %s ', name_database)
            imagelist, edgeslist, MSE = self.process(exp, name_database=name_database, noise=noise)
#            edgeslist_db.append(edgeslist)
            if not(imagelist == 'locked'):
                self.log.info(' >> computing histogram for %s ', name_database)
                try:
                    v_hist_ = np.load(matname + '_kmeans_hist.npy')
                except Exception as e:
                    self.log.info(' >> There is no histogram, computing: %s ', e)
                    # images are separated in a learning and classification set: the histogram is computed on the first half of the images
                    N_image = edgeslist.shape[2]
                    v_hist_ = self.cohistedges(edgeslist[:, :, :N_image//2], display=None)
                    np.save(matname + '_kmeans_hist.npy', v_hist_)
            else:
                self.log.info('XX The process extracting edges in %s  is locked', name_database)
                locked = True
            if not(locked):
                # we store the histogram for the first half of the image for each class
                edgeslist_db.append(edgeslist)
                if do_scale:
                    v_hist.append(v_hist_)
                else:
                    v_hist.append(v_hist_.sum(axis=3))

        exp = '.pdf'
        figname = os.path.join(self.pe.figpath, exp + '_KL_' + databases[0] + '_' + databases[1] + ext)
        if not(os.path.isfile(figname)) and not(locked):
            rho = []
            for i_database, name_database in enumerate(databases):
                N_image = edgeslist_db[i_database].shape[2]
                N_image_half = int(np.ceil(N_image/2.))
                rho_ = np.zeros((N_image_half,)) # we test on the other half of the edges
                for ii_image, i_image in enumerate(range(N_image-N_image_half, N_image)):
                    v_hist_obs = self.cohistedges(edgeslist_db[i_database][:, :, i_image][..., np.newaxis], display=None)
                    if not(do_scale): v_hist_obs = v_hist_obs.sum(axis=3)
                    d_A = KL(v_hist_obs, v_hist[0])
                    d_B = KL(v_hist_obs, v_hist[1])
                    if geometric: rho_[ii_image] = d_A/np.sqrt(d_A**2+d_B**2)
                    else: rho_[ii_image] = d_A/np.sqrt(d_A*d_B)#/(d_A+d_B)
                rho.append(rho_)
            ha, dump = np.histogram(rho[0], np.linspace(0, 1., rho_quant), normed=True)#, density=True)
            hb, dump = np.histogram(rho[1], np.linspace(0, 1., rho_quant), normed=True)#, density=True)
            ha /= np.sum(ha)
            hb /= np.sum(hb)
            cdfa, cdfb = np.cumsum(ha), np.cumsum(hb)
            # plots
            fig, ax = plt.subplots(figsize=(self.pe.figsize_hist, self.pe.figsize_hist))
            ax.plot(cdfb, cdfa, color='r', lw=2)
            ax.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), 'k--', lw=2)
            print(" >> AUC for experiment ", exp, " classifiying between databases ", databases, " = ", AUC(cdfb, cdfa))
            ax.set_xlabel('false positive rate = 1 - Specificity')
            ax.set_ylabel('false negative rate = Sensitivity')
            ax.set_axis('tight')
            ax.text(0.5, 0.1, 'AUC = ' + str(AUC(cdfb, cdfa)))
            self.savefig(fig, figname)


def AUC(cdfb, cdfa):
    """
    Given two CDF curves, returns the area under the curve they define.

    We use the trapezoidal approximation.
    http://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_Under_Curve

    The area under ROC curve specifies the probability that, when we draw
    one positive and one negative example at random, the decision function
    assigns a higher value to the positive than to the negative example.

    """
    return np.sum((cdfb[1:]-cdfb[:-1])*(cdfa[1:]+cdfa[:-1])/2)


def _test():
    import doctest
    doctest.testmod()
#####################################
#
if __name__ == '__main__':
    _test()

    #### Main
    """
    Some examples of use for the class

    """
    print('main')
#     from plt import imread
#     # whitening
#     image = imread('database/gris512.png')[:,:,0]
#     lg = LogGabor(image.shape)
