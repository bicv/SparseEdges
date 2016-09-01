# -*- coding: utf8 -*-
from __future__ import division, print_function
"""
SparseEdges

See http://pythonhosted.org/SparseEdges

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
__docformat__ = "restructuredtext"
import numpy as np
import os
PID, HOST = os.getpid(), os.uname()[1]
TAG = 'host-' + HOST + '_pid-' + str(PID)
# -------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
import sys, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hashlib
import pickle
from scipy.stats import powerlaw

from LogGabor import LogGabor

class SparseEdges(LogGabor):
    def __init__(self, pe):
        """
        Initializes the SparseEdges class

        """
        LogGabor.__init__(self, pe)
        self.init()
        self.init_logging(name='SparseEdges')

    def init(self):
        LogGabor.init(self)

        self.n_levels = int(np.log(np.max((self.pe.N_X, self.pe.N_Y)))/np.log(self.pe.base_levels))
        self.sf_0 = 1. / np.logspace(1, self.n_levels, self.n_levels, base=self.pe.base_levels)
        self.theta = np.linspace(-np.pi/2, np.pi/2, self.pe.n_theta+1)[1:]

        self.oc = (self.pe.N_X * self.pe.N_Y * self.pe.n_theta * self.n_levels) #(1 - self.pe.base_levels**-2)**-1)

    def run_mp(self, image, verbose=False, progress=False):
        """
        runs the MatchingPursuit algorithm on image

        """
        edges = np.zeros((6, self.pe.N))
        image_ = image.copy()
#         residual = image.copy()
#         RMSE = np.ones(self.pe.N)
        if self.pe.do_whitening: image_ = self.whitening(image_)
        C = self.init_C(image_)
        if progress:
            import pyprind
            my_prbar = pyprind.ProgPercent(self.pe.N)   # 1) initialization with number of iterations
        for i_edge in range(self.pe.N):
#             RMSE[i_edge] = np.sum((residual - image_)**2)
            # MATCHING
            ind_edge_star = self.argmax(C)
            if not (self.pe.MP_rho is None):
                print('dooh!')
                if i_edge==0: C_Max = np.absolute(C[ind_edge_star])
                coeff = self.pe.MP_alpha * (self.pe.MP_rho ** i_edge) * C_Max
                # recording
                if verbose: print('Edge ', i_edge, '/', self.pe.N, ' - Max activity (quant mode) : ', np.absolute(C[ind_edge_star]), ', coeff/alpha=', coeff/self.pe.MP_alpha , ' phase= ', np.angle(C[ind_edge_star], deg=True), ' deg,  @ ', ind_edge_star)
            elif self.pe.MP_alpha == np.inf:
                # linear coding - no sparse coding, see ``backprop`` function
                coeff = np.absolute(C[ind_edge_star])
            else:
                coeff = self.pe.MP_alpha * np.abs(C[ind_edge_star])
                # recording
                if verbose: print('Edge ', i_edge, '/', self.pe.N, ' - Max activity  : ', np.absolute(C[ind_edge_star]), ' phase= ', np.angle(C[ind_edge_star], deg=True), ' deg,  @ ', ind_edge_star)
            if progress: my_prbar.update()
            edges[:, i_edge] = np.array([ind_edge_star[0]*1., ind_edge_star[1]*1.,
                                         self.theta[ind_edge_star[2]],
                                         self.sf_0[ind_edge_star[3]],
                                         coeff, np.angle(C[ind_edge_star])])
            # PURSUIT
            C = self.backprop(C, ind_edge_star)
        return edges, C

    def init_C(self, image):
        C = np.empty((self.pe.N_X, self.pe.N_Y, self.pe.n_theta, self.n_levels), dtype=np.complex)
        for i_sf_0, sf_0 in enumerate(self.sf_0):
            for i_theta, theta in enumerate(self.theta):
                FT_lg = self.loggabor(0, 0, sf_0=sf_0, B_sf=self.pe.B_sf,
                                    theta=theta, B_theta=self.pe.B_theta)
                C[:, :, i_theta, i_sf_0] = self.FTfilter(image, FT_lg, full=True)
        return C

    def argmax(self, C):
        """
        Returns the ArgMax from C by returning the
        (x_pos, y_pos, theta, scale)  tuple

        >>> C = np.random.randn(10, 10, 5, 4)
        >>> x_pos, y_pos, theta, scale = mp.argmax(C)
        >>> C[x_pos][y_pos][theta][scale] = C.max()

        """
        ind = np.absolute(C).argmax()
        return np.unravel_index(ind, C.shape)

    def backprop(self, C, ind_edge_star):
        """
        Removes edge_star from the activity

        """
        if self.pe.MP_alpha == np.inf:
            # linear coding - no sparse coding
            C[ind_edge_star] = 0.
        else:
            C_star = self.pe.MP_alpha * C[ind_edge_star]
            FT_lg_star = self.loggabor(ind_edge_star[0]*1., ind_edge_star[1]*1.,
                                          theta=self.theta[ind_edge_star[2]], B_theta=self.pe.B_theta,
                                          sf_0=self.sf_0[ind_edge_star[3]], B_sf=self.pe.B_sf,
                                          )
            # image of the winning filter
            lg_star = self.invert(C_star*FT_lg_star, full=False)
            for i_sf_0, sf_0 in enumerate(self.sf_0):
                for i_theta, theta in enumerate(self.theta):
                    FT_lg = self.loggabor(0, 0, sf_0=sf_0, B_sf=self.pe.B_sf, theta=theta, B_theta=self.pe.B_theta)
                    C[:, :, i_theta, i_sf_0] -= self.FTfilter(lg_star, FT_lg, full=True)
        return C

    def reconstruct(self, edges, mask=False):
        image = np.zeros((self.pe.N_X, self.pe.N_Y))
#        print edges.shape, edges[:, 0]
        for i_edge in range(edges.shape[1]):#self.pe.N):
            # TODO : check that it is correct when we remove alpha when making new MP
            if not mask or ((edges[0, i_edge]/self.pe.N_X -.5)**2+(edges[1, i_edge]/self.pe.N_Y -.5)**2) < .5**2:
                image += self.invert(edges[4, i_edge] * np.exp(1j*edges[5, i_edge]) *
                                    self.loggabor(
                                                    edges[0, i_edge], edges[1, i_edge],
                                                    theta=edges[2, i_edge], B_theta=self.pe.B_theta,
                                                    sf_0=edges[3, i_edge], B_sf=self.pe.B_sf,
                                                    ),
                                    full=False)
        return image

    def adapt(self, edges):
        # TODO : implement a COMP adaptation of the thetas and scales tesselation of Fourier space
        pass

    def show_edges(self, edges, fig=None, a=None, image=None, norm=True,
                   color='auto', v_min=-1., v_max=1., show_phase=True, gamma=1.,
                   pedestal=0., mask=False, mappable=False):
        """
        Shows the quiver plot of a set of edges, optionally associated to an image.

        """
        import matplotlib.cm as cm
        if fig==None:
            #  Figure :                      height         ----------           width
            fig = plt.figure(figsize=(self.pe.figsize_edges*self.pe.N_Y/self.pe.N_X, self.pe.figsize_edges))
        if a==None:
            border = 0.0
            a = fig.add_axes((border, border, 1.-2*border, 1.-2*border), axisbg='w')
        a.axis(c='b', lw=0, frame_on=False)

        # HACK
        if color == 'black' or color == 'redblue' or color in['brown', 'green', 'blue']: #cocir or chevrons
            linewidth = self.pe.line_width_chevrons
            scale = self.pe.scale_chevrons
        else:
            linewidth = self.pe.line_width
            scale = self.pe.scale

        opts= {'extent': (0, self.pe.N_Y, self.pe.N_X, 0), # None, #
               'cmap': cm.gray,
               'vmin':v_min, 'vmax':v_max, 'interpolation':'nearest', 'origin':'upper'}
#         origin : [‘upper’ | ‘lower’], optional, default: None
#         Place the [0,0] index of the array in the upper left or lower left corner of the axes. If None, default to rc image.origin.
#         extent : scalars (left, right, bottom, top), optional, default: None
#         Data limits for the axes. The default assigns zero-based row, column indices to the x, y centers of the pixels.
        if type(image)==np.ndarray:
#             if image.ndim==2: opts['cmap'] = cm.gray
            if norm: image = self.normalize(image, center=True, use_max=True)
            a.imshow(image, **opts)
        else:
            a.imshow([[v_max]], **opts)
        if edges.shape[1] > 0:
            from matplotlib.collections import LineCollection, PatchCollection
            import matplotlib.patches as patches
            # draw the segments
            segments, colors, linewidths = list(), list(), list()
            patch_circles = []

            X, Y, Theta, Sf_0 = edges[1, :]+.5, edges[0, :]+.5, np.pi -  edges[2, :], edges[3, :] # HACK in orientation
#             X, Y, Theta, Sf_0 = edges[1, :]+.5, self.pe.N_X - edges[0, :]-.5, edges[2, :], edges[3, :] # HACK in orientation
            weights = edges[4, :]
            weights = weights/(np.abs(weights)).max()
            phases = edges[5, :]

            for x, y, theta, sf_0, weight, phase in zip(X, Y, Theta, Sf_0, weights, phases):
                if not mask or ((x/self.pe.N_X -.5)**2+(y/self.pe.N_Y -.5)**2) < .5**2:
                    u_, v_ = np.cos(theta)*scale/sf_0, np.sin(theta)*scale/sf_0
                    segment = [(x - u_, y - v_), (x + u_, y + v_)]
                    segments.append(segment)
                    if color=='auto':
                        if not(show_phase):
                            fc = cm.hsv(0, alpha=pedestal + (1. - pedestal)*weight**gamma)
                        else:
                            fc = cm.hsv((phase/np.pi/2) % 1., alpha=pedestal + (1. - pedestal)*weight**gamma)
# TODO                     https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/
#                     RGB_weight = [0.299, 0.587, 0.114]
#                     luminance = np.sqrt(np.dot(np.array(fc[:, :3]) ** 2, RGB_weight))
#                     print luminance
#                     fc[:, :3] /= luminance

                    elif color == 'black':
                        fc = (0, 0, 0, 1)# black
                    elif color == 'green': # figure 1DE
                        fc = (0.05, 0.5, 0.05, np.abs(weight)**gamma)
                    elif color == 'blue': # figure 1DE
                        fc = (0.05, 0.05, 0.5, np.abs(weight)**gamma)
                    elif color == 'brown': # figure 1DE
                        fc = (0.5, 0.05, 0.05, np.abs(weight)**gamma)
                    else:
                        fc = ((np.sign(weight)+1)/2, 0, (1-np.sign(weight))/2, np.abs(weight)**gamma)
                    colors.append(fc)
                    linewidths.append(linewidth) # *weight thinning byalpha...
                    patch_circles.append(patches.Circle((x,y), self.pe.scale_circle*scale/sf_0, lw=0., facecolor=fc, edgecolor='none'))

            line_segments = LineCollection(segments, linewidths=linewidths, colors=colors, linestyles='solid')
            a.add_collection(line_segments)
            circles = PatchCollection(patch_circles, match_original=True)
            a.add_collection(circles)

        if True: # HACK not(color=='auto'):# chevrons maps etc...
            plt.setp(a, xticks=[])
            plt.setp(a, yticks=[])

        if mask:
            linewidth_mask = 1 # HACK
            circ = plt.Circle((.5*self.pe.N_Y, .5*self.pe.N_Y), radius=0.5*self.pe.N_Y-linewidth_mask/2., fill=False, facecolor='none', edgecolor = 'black', alpha = 0.5, ls='dashed', lw=linewidth_mask)
            a.add_patch(circ)
        a.axis([0, self.pe.N_Y, self.pe.N_X, 0])
        a.grid(b=False, which="both")
        plt.draw()
        if mappable:
            return fig, a, line_segments
        else:
            return fig, a

    def texture(self, N_edge=256, a=None, filename='', croparea='', randn=True):
        # a way to get always the same seed for each image
        if not (filename==''):# or not (croparea==''):
            np.random.seed(seed=int(int("0x" +  hashlib.sha224((filename+str(croparea)).encode('utf-8')).hexdigest(), 0)*1. % 4294967295))
        # white noise or texture
        if randn:
            return np.random.randn(self.pe.N_X, self.pe.N_Y)
        else:
            edgeslist = np.zeros((6, N_edge))
            edgeslist[0, :] = self.pe.N_X * np.random.rand(N_edge)
            edgeslist[1, :] = self.pe.N_X * np.random.rand(N_edge)
            edgeslist[2, :] = (np.pi* np.random.rand(N_edge) ) % np.pi
            if a is None:
                edgeslist[3, :] =  self.sf_0[np.random.randint(self.sf_0.size, size=(N_edge))] # best would be to have more high frequency components
                edgeslist[4, :] = np.random.randn(N_edge)
            else:
                #edgeslist[4, :] = 1 / np.random.power(a=a, size=N_edge)
                edgeslist[3, :] =  self.sf_0.max() * powerlaw.rvs(a=4., size = N_edge) # HACK
                edgeslist[4, :]  = np.random.pareto(a=a, size=(N_edge)) + 1


            edgeslist[5, :] = 2*np.pi*np.random.rand(N_edge)
            image_rec = self.reconstruct(edgeslist)
            image_rec /= image_rec.std()
            return image_rec
    #  TODO : use MotionClouds strategy
    # for i_sf, sf_0_ in enumerate(sf_0 * scaling ** np.linspace(-1, 1, n_sf)):
    #     z = mc.envelope_gabor(fx, fy, ft, V_X=V_X, V_Y=V_Y, B_V=B_V, sf_0=sf_0_, B_sf=B_sf*sf_0_/sf_0, B_theta=B_theta)
    #     texton = mc.random_cloud(z, impulse=True) # TODO : and the seed?
    #     if verbose:
    #         print(' ⇒ At scale ', sf_0_, ', the texton has energy ', np.sqrt((texton**2).mean()),
    #               ', the number of components is ', int(mask.sum()))
    #
    #     Fz = np.fft.fftn(( events[:, :, :, i_sf] ))
    #     Fz = np.fft.fftshift(Fz)
    #     Fz *= z
    #     Fz = np.fft.ifftshift(Fz)
    #     Fz[0, 0, 0] = 0. # removing the DC component
    #     droplets_mc += np.fft.ifftn((Fz)).real
    # return events, droplets_mc

    def full_run(self, exp, name_database, imagelist, noise, N_do=2, time_sleep=.1):
        """
        runs the edge extraction for a list of images

        """

        for path in self.pe.figpath, self.pe.matpath, self.pe.edgefigpath, self.pe.edgematpath:
            if not(os.path.isdir(path)): os.mkdir(path)
        for _ in range(N_do): # repeat this loop to make sure to scan everything
            global_lock = False # will switch to True when we resume a batch and detect that one edgelist is not finished in another process
            for index in np.random.permutation(np.arange(len(imagelist))):
                filename, croparea = imagelist[index]
#                 signal = do_edge(self, image, exp, name_database, filename, croparea)
#                         def do_edge(self, image, exp, name_database, filename, croparea):
                path = os.path.join(self.pe.edgematpath, exp + '_' + name_database)
                if not(os.path.isdir(path)): os.mkdir(path)
                matname = os.path.join(path, filename + str(croparea) + '.npy')
                if not(os.path.isfile(matname)):
                    time.sleep(time_sleep*np.random.rand())
                    if not(os.path.isfile(matname + '_lock')):
                        self.log.info('Doing edge extraction of %s ', matname)
                        open(matname + '_lock', 'w').close()
                        image, filename_, croparea_ = self.patch(name_database, filename=filename, croparea=croparea)
                        if noise > 0.: image += noise*image[:].std()*self.texture(filename=filename, croparea=croparea)
                        edges, C = self.run_mp(image, verbose=self.pe.verbose>50)
                        np.save(matname, edges)
                        self.log.info('Finished edge extraction of %s ', matname)
                        try:
                            os.remove(matname + '_lock')
                        except Exception as e:
                            self.log.error('Failed to remove lock file %s_lock, error : %s ', matname, traceback.print_tb(sys.exc_info()[2]))
                    else:
                        self.log.info('The edge extraction at step %s is locked', matname)
                        global_lock = True
        if global_lock is True:
            self.log.error(' some locked edge extractions ')
            return 'locked'
        else:
            try:
                N_image = len(imagelist)
                edgeslist = np.zeros((6, self.pe.N, N_image))
                i_image = 0
                for filename, croparea in imagelist:
                    matname = os.path.join(self.pe.edgematpath, exp + '_' + name_database, filename + str(croparea) + '.npy')
                    edgeslist[:, :, i_image] = np.load(matname)
                    i_image += 1
                return edgeslist
            except Exception as e:
                self.log.error(' some locked edge extractions %s, error on file %s', e, matname)
                return 'locked'

    def full_RMSE(self, exp, name_database, imagelist):
        N_do = 2
        for _ in range(N_do): # repeat this loop to make sure to scan everything
            global_lock = False # will switch to True when we resume a batch and detect that one edgelist is not finished in another process
            for filename, croparea in imagelist:
                matname = os.path.join(self.pe.edgematpath, exp + '_' + name_database, filename + str(croparea) + '_RMSE.npy')
                if not(os.path.isfile(matname)):
                    if not(os.path.isfile(matname + '_lock')):
                        open(matname + '_lock', 'w').close()
                        image, filename_, croparea_ = self.patch(name_database, filename=filename, croparea=croparea)
                        edges = np.load(os.path.join(self.pe.edgematpath, exp + '_' + name_database, filename + str(croparea) + '.npy'))
                        # computing RMSE
                        RMSE = np.ones(self.pe.N)
                        if self.pe.MP_alpha == np.inf:
                            RMSE *= np.nan
                        else:
                            image_ = image.copy()
                            image_rec = np.zeros_like(image_)
                            if self.pe.do_whitening: image_ = self.whitening(image_)
                            for i_N in range(self.pe.N):
                                image_rec += self.reconstruct(edges[:, i_N][:, np.newaxis], mask=self.pe.do_mask)
                                RMSE[i_N] =  ((image_-image_rec)**2).sum()
                        np.save(matname, RMSE)
                        try:
                            os.remove(matname + '_lock')
                        except Exception as e:
                            self.log.error('Failed to remove lock file %s_lock, error : %s ', matname, traceback.print_tb(sys.exc_info()[2]))
                    else:
                        self.log.info('The edge extraction at step %s is locked', matname)
                        global_lock = True
        if global_lock is True:
            self.log.error(' some locked RMSE extractions ')
            return 'locked'
        else:
            try:
                N_image = len(imagelist)
                RMSE = np.ones((N_image, self.pe.N))
                for i_image in range(N_image):
                    filename, croparea = imagelist[i_image]
                    matname_RMSE = os.path.join(self.pe.edgematpath, exp + '_' + name_database, filename + str(croparea) + '_RMSE.npy')
                    RMSE[i_image, :] = np.load(matname_RMSE)
                return RMSE
            except Exception as e:
                self.log.error(' some locked RMSE extractions %s, error ', e)
                return 'locked'

    def init_edges(self):
        # configuring histograms
        # sequence of scalars,it defines the bin edges, including the rightmost edge.
        self.edges_d = np.linspace(self.pe.d_min, self.pe.d_max, self.pe.N_r+1)
        self.edges_phi = np.linspace(-np.pi/2, np.pi/2, self.pe.N_phi+1) + np.pi/self.pe.N_phi/2
        self.edges_theta = np.linspace(-np.pi/2, np.pi/2, self.pe.N_Dtheta+1) + np.pi/self.pe.N_Dtheta/2
        self.edges_sf_0 = 2**np.arange(np.ceil(np.log2(self.pe.N_X)))
        self.edges_loglevel = np.linspace(-self.pe.loglevel_max, self.pe.loglevel_max, self.pe.N_scale+1)

    def histedges_theta(self, edgeslist, fig=None, a=None, display=True):
        """
        First-order stats

        p(theta | I )

        """
        self.init_edges()

        theta = (edgeslist[2, ...].ravel())
        theta = ((theta + np.pi/2 - np.pi/self.pe.N_Dtheta/2)  % np.pi ) - np.pi/2  + np.pi/self.pe.N_Dtheta/2
        value = edgeslist[4, ...].ravel()

        if self.pe.edge_mask:
            # remove edges whose center position is not on the central disk
            x , y = edgeslist[0, ...].ravel().real, edgeslist[1, ...].ravel().real
            mask = ((x/self.pe.N_X -.5)**2+(y/self.pe.N_Y -.5)**2) < .5**2
            theta = theta[mask]
            value = value[mask]

#         print theta.min(), theta.max(),
        weights = np.absolute(value)/(np.absolute(value)).sum()
        theta_bin = self.edges_theta # np.hstack((self.theta, self.theta[0]+np.pi))  + np.pi/self.pe.N_Dtheta/2
#         print theta_bin.min(), theta_bin.max()
        v_hist, v_theta_edges_ = np.histogram(theta, bins=theta_bin, density=True, weights=weights)
        v_hist /= v_hist.sum()
        if display:
            if fig==None: fig = plt.figure(figsize=(self.pe.figsize_hist, self.pe.figsize_hist))
            if a==None: a = plt.axes(polar=True, axisbg='w')
#             see http://blog.invibe.net/posts/14-12-09-polar-bar-plots.html
            a.bar(theta_bin[1:], np.sqrt(v_hist), width=theta_bin[:-1] - theta_bin[1:], color='#66c0b7')# edgecolor="none")
            a.bar(theta_bin[1:]+np.pi, np.sqrt(v_hist), width=theta_bin[:-1] - theta_bin[1:], color='#32ab9f')
            plt.setp(a, yticks=[])
            return fig, a
        else:
            return v_hist, v_theta_edges_

    def histedges_scale(self, edgeslist, fig=None, a=None, display=True):
        """
        First-order stats for the scale

        p(scale | I )

        """
        self.init_edges()

        sf_0 = (edgeslist[3, ...].ravel())
        value = edgeslist[4, ...].ravel()
        if self.pe.edge_mask:
            # remove edges whose center position is not on the central disk
            x , y = edgeslist[0, ...].ravel().real, edgeslist[1, ...].ravel().real
            mask = ((x/self.pe.N_X -.5)**2+(y/self.pe.N_Y -.5)**2) < .5**2
            sf_0 = sf_0[mask]
            value = value[mask]

        weights = np.absolute(value)/(np.absolute(value)).sum()
        v_hist, v_sf_0_edges_ = np.histogram(sf_0, self.edges_sf_0, density=True, weights=weights)
        v_hist /= v_hist.sum()
        if display:
            if fig==None: fig = plt.figure(figsize=(self.pe.figsize_hist, self.pe.figsize_hist))
            if a==None: a = fig.add_subplot(111, axisbg='w')
            a.bar(v_sf_0_edges_[:-1], v_hist)
            plt.setp(a, yticks=[])
            plt.xlabel(r'$sf_0$')
            plt.ylabel('probability')
            return fig,a
        else:
            return v_hist, v_theta_edges_

    def cohistedges(self, edgeslist, v_hist=None, prior=None,
                    fig=None, a=None, symmetry=True,
                    display='chevrons', v_min=None, v_max=None, labels=True, mappable=False, radius=None,
                    xticks=False, half=False, dolog=True, color='redblue', colorbar=True, cbar_label=True):
        """
        second-order stats= center all edges around the current one by rotating and scaling

        p(x-x_, y-y_, theta-theta_ | I, x_, y_, theta_)

        """
        self.init_edges()

        if not(edgeslist is None):
            v_hist = None
            six, N_edge, N_image = edgeslist.shape
            for i_image in range(N_image):
                # retrieve individual positions, orientations, scales and coefficients
                X, Y = edgeslist[0, :, i_image], edgeslist[1, :, i_image]
                Theta = edgeslist[2, :, i_image]
                Sf_0 = edgeslist[3, :, i_image]
                value = edgeslist[4, :, i_image]
                phase = edgeslist[5, :, i_image]
                if self.pe.edge_mask:
                    # remove edges whose center position is not on the central disk
                    mask = ((X/self.pe.N_X -.5)**2+(Y/self.pe.N_Y -.5)**2) < .5**2
                    X = X[mask]
                    Y = Y[mask]
                    Theta = Theta[mask]
                    Sf_0 = Sf_0[mask]
                    value = value[mask]
                    phase = phase[mask]

                # TODO: should we normalize weights by the max (while some images are "weak")? the corr coeff would be an alternate solution... / or simply the rank
                Weights = value # np.absolute(value)#/(np.absolute(value)).sum()
                if self.pe.do_rank: Weights[Weights.argsort()] = np.linspace(1./Weights.size, 1., Weights.size)
                # TODO: include phases or use that to modify center of the edge
                # TODO: or at least on the value (ON or OFF) of the edge
                # TODO: normalize weights by their relative order to be independent of the texture

                # to raise on numerical error, issue
                np.seterr(all='ignore')
                dx = X[:, np.newaxis] - X[np.newaxis, :]
                dy = Y[:, np.newaxis] - Y[np.newaxis, :]
                # TODO : make an histogram on log-radial coordinates and theta versus scale
                d = np.sqrt(dx**2 + dy**2) / self.pe.N_X  # distance normalized by the image size
                # TODO: check that we correctly normalize position by the scale of the current edge
                if self.pe.scale_invariant: d *= np.sqrt(Sf_0[:, np.newaxis]*Sf_0[np.newaxis, :])#*np.sqrt(self.pe.N_X)
                d *= self.pe.d_width # distance in visual angle
                theta = Theta[:, np.newaxis] - Theta[np.newaxis, :]
                phi = np.arctan2(dy, dx) - np.pi/2 - Theta[np.newaxis, :]
                if symmetry: phi -= theta/2
                loglevel = np.log2(Sf_0[:, np.newaxis]) - np.log2(Sf_0[np.newaxis, :])
                weights = Weights[:, np.newaxis] * Weights[np.newaxis, :]
                if self.pe.weight_by_distance:
                    # normalize weights by the relative distance (bin areas increase with radius)
                    # it makes sense to give less weight to "far bins"
                    weights /= (d + 1.e-6) # warning, some are still at the same position d=0...
                # exclude self-occurence
#                 weights[np.diag_indices_from(weights)] = 0.
                np.fill_diagonal(weights, 0.)
                # TODO check: if not self.pe.multiscale: weights *= (Sf_0[:, np.newaxis]==Sf_0[inp.newaxis, :])
                # just checking if we get different results when selecting edges with a similar phase (von Mises profile)
                if self.pe.kappa_phase>0:
                    # TODO: should we use the phase information to refine position?
                    # https://en.wikipedia.org/wiki/Atan2
                    weights *= np.exp(self.pe.kappa_phase*np.cos(np.arctan2(phase[:, np.newaxis], phase[np.newaxis, :])))

                if weights.sum()>0:
                    weights /= weights.sum()
                    weights = weights.ravel()
                else:
                    weights = np.ones_like(weights)
#                 print 'd', d.min(), self.edges_d.min(), ' / ', d.max(), self.edges_d.max(), ' / ', d.std(), ' / ', np.median(d), ' / ', (d*weights).sum(), ' / ', weights.sum()

#                 print (np.sin(theta + theta.T)).std()
#                 print (np.sin(phi - phi.T)).std()
#                 print 'phi', phi.min(), self.edges_phi.min(), ' / ', phi.max(), self.edges_phi.max()
#                 print 'theta', theta.min(), self.edges_theta.min(), ' / ', theta.max(), self.edges_theta.max()
                # putting everything in the right range:
                phi = ((phi + np.pi/2  - np.pi/self.pe.N_phi/2 ) % (np.pi)) - np.pi/2  + np.pi/self.pe.N_phi/2
                theta = ((theta + np.pi/2 - np.pi/self.pe.n_theta/2)  % (np.pi) ) - np.pi/2  + np.pi/self.pe.n_theta/2
#                 print 'phi', phi.min() - self.edges_phi.min(), ' / ', phi.max() - self.edges_phi.max()
#                 print 'theta', theta.min() - self.edges_theta.min(), ' / ', theta.max() - self.edges_theta.max()
                v_hist_, edges_ = np.histogramdd([d.ravel(), phi.ravel(), theta.ravel(), loglevel.ravel()], #data,
                                                 bins=(self.edges_d, self.edges_phi, self.edges_theta, self.edges_loglevel),
                                                 normed=False, # TODO check if correct True,
                                                 weights = weights
                                                )
#                 print v_hist_.sum(), v_hist_.min(), v_hist_.max(), d.ravel().shape
                if v_hist_.sum()<.01: self.log.debug(' less than 1 percent of co-occurences within ranges: %f ', v_hist_.sum())
                if not(v_hist_.sum() == 0.):
                    # add to the full histogram
                    if v_hist is None:
                        v_hist = v_hist_*1.
                    else:
                        v_hist += v_hist_*1.
        if v_hist is None: # or (v_hist.sum() == 0.):
            v_hist = np.ones(v_hist_.shape)

        v_hist /= v_hist.sum()

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        if display=='full':
            if fig==None:
                fig = plt.figure(figsize=(self.pe.figsize_cohist, self.pe.figsize_cohist))
            options = {'cmap': cm.jet, 'interpolation':'nearest', 'vmin':0., 'origin': 'lower'}
            a1 = fig.add_subplot(221, axisbg='w')#, polar = True)
            a1.imshow((v_hist.sum(axis=3).sum(axis=2)), **options)
            if symmetry:
                a1.set_xlabel('psi')
            else:
                a1.set_xlabel('phi')
            a1.set_xticks([0., self.edges_phi.size/2. -1.5, self.edges_phi.size-2.])
            a1.set_xticklabels(['-pi/2 + bw ', '0', 'pi/2'])
            a1.set_ylabel('d')
            edges_d_half = .5*(self.edges_d[1:] + self.edges_d[:-1])
            a1.set_yticks([0., self.edges_d.size-2.])
            a1.set_yticklabels([str(edges_d_half[0]), str(edges_d_half[-1])])
            a1.axis('tight')
            a2 = fig.add_subplot(222, axisbg='w')#edges_[0], edges_[2],
            a2.imshow((v_hist.sum(axis=3).sum(axis=1)), **options)
            a2.set_xlabel('theta')
            a2.set_xticks([0., self.edges_theta.size/2.-1.5, self.edges_theta.size-2.])
            a2.set_xticklabels(['-pi/2 + bw', '0', 'pi/2'])
            a2.set_ylabel('d')
            a2.set_yticks([0., self.edges_d.size-2.])
            a2.set_yticklabels([str(edges_d_half[0]), str(edges_d_half[-1])])
            a2.axis('tight')
            a3 = fig.add_subplot(223, axisbg='w')#edges_[1], edges_[2],
            a3.imshow((v_hist.sum(axis=3).sum(axis=0)).T, **options)
            if symmetry:
                a3.set_xlabel('psi')
            else:
                a3.set_xlabel('phi')
            a3.set_xticks([0., self.edges_phi.size/2. - 1.5, self.edges_phi.size-2.])
            a3.set_xticklabels(['-pi/2 + bw', '0', 'pi/2'])
            a3.set_ylabel('theta')
            a3.set_yticks([0., self.edges_theta.size/2. - 1.5, self.edges_theta.size-2.])
            a3.set_yticklabels(['-pi/2 + bw', '0', 'pi/2'])
            a3.axis('tight')
            a4 = fig.add_subplot(224, axisbg='w')#, polar = True)
            a4.imshow((v_hist.sum(axis=1).sum(axis=1)), **options)
            a4.set_xlabel('levels')
            a4.set_xticks([0., self.pe.N_scale/2. -.5, self.pe.N_scale -1.])
            a4.set_xticklabels(['smaller', '0', 'bigger'])
            a4.set_ylabel('d')
            a4.set_yticks([0., self.edges_d.size-2.])
            a4.set_yticklabels([str(edges_d_half[0]), str(edges_d_half[-1])])
            a4.axis('tight')
#             plt.tight_layout()
            return fig, a1, a2, a3, a4

        elif display=='colin_geisler':
            edge_scale = 64.
            try:
                if fig==None:
                    fig = plt.figure(figsize=(self.pe.figsize_cohist, self.pe.figsize_cohist))
                    if a==None:
                        a = fig.add_subplot(111)
                v_hist_noscale = v_hist.sum(axis=3)
                colin_edgelist = np.zeros((6, self.pe.N_r * self.pe.N_phi * 2 + 1 ))
                colin_argmax = np.argmax(v_hist_noscale, axis=2)
                for i_r, d_r in enumerate(self.edges_d[:-1]):
                    for i_phi, phi in enumerate(self.edges_phi[:-1]):
                        rad = d_r / self.pe.d_max * max(self.pe.N_X, self.pe.N_Y) /2
                        ii_phi = i_r * self.pe.N_phi
                        colin_edgelist[0:2, ii_phi + i_phi] =  self.pe.N_X /2 - rad * np.sin(phi + np.pi/self.pe.N_phi/2), self.pe.N_Y /2 + rad * np.cos(phi + np.pi/self.pe.N_phi/2)
                        colin_edgelist[2, ii_phi + i_phi] = self.edges_theta[colin_argmax[i_r, i_phi]] + np.pi/self.pe.N_Dtheta/2
                        colin_edgelist[3, ii_phi + i_phi] = edge_scale
                        colin_edgelist[4, ii_phi + i_phi] = v_hist_noscale[i_r, i_phi, colin_argmax[i_r, i_phi]]
                        # symmetric
                        colin_edgelist[:, ii_phi + i_phi +  self.pe.N_r * self.pe.N_phi] = colin_edgelist[:, ii_phi + i_phi]
                        colin_edgelist[0:2, ii_phi + i_phi +  self.pe.N_r * self.pe.N_phi] = self.pe.N_X - colin_edgelist[0, ii_phi + i_phi], self.pe.N_Y - colin_edgelist[1, ii_phi + i_phi]
                # reference angle
                colin_edgelist[:, -1] = [self.pe.N_X /2, self.pe.N_Y /2, 0, edge_scale, colin_edgelist[4,:].max() *1.2, 0.]
                return self.show_edges(colin_edgelist, fig=fig, a=a, image=None, v_min=0., v_max=v_hist_noscale.max(), color=color)
            except Exception as e:
                self.log.error(' failed to generate colin_geisler plot, %s', traceback.print_tb(sys.exc_info()[2]))
                return e, None # HACK to return something instead of None

        elif display=='cocir_geisler':
            edge_scale = 64.
            try:
                if fig==None:
                    fig = plt.figure(figsize=(self.pe.figsize_cohist, self.pe.figsize_cohist))
                    if a==None:
                        a = fig.add_subplot(111)
                v_hist_noscale = v_hist.sum(axis=3)
                cocir_edgelist = np.zeros((6, self.pe.N_r * self.pe.N_Dtheta * 2 + 1 ))
                cocir_proba = np.argmax(v_hist_noscale, axis=1)
                for i_r, d_r in enumerate(self.edges_d[:-1]):
                    for i_theta, theta in enumerate(self.edges_theta[:-1]):
                        rad = d_r / self.pe.d_max * max(self.pe.N_X, self.pe.N_Y) /2
                        ii_theta = i_r * self.pe.N_Dtheta
                        cocir_edgelist[0:2, ii_theta + i_theta] =  self.pe.N_X /2 - rad * np.sin( self.edges_phi[cocir_proba[i_r, i_theta]] + np.pi/self.pe.N_phi/2), self.pe.N_Y /2 + rad * np.cos( self.edges_phi[cocir_proba[i_r, i_theta]] + np.pi/self.pe.N_phi/2)
                        cocir_edgelist[2, ii_theta + i_theta] = theta + np.pi/self.pe.N_Dtheta/2
                        cocir_edgelist[3, ii_theta + i_theta] = edge_scale
                        cocir_edgelist[4, ii_theta + i_theta] = v_hist_noscale[i_r, cocir_proba[i_r, i_theta], i_theta]
                        # symmetric
                        cocir_edgelist[:, ii_theta + i_theta +  self.pe.N_r * self.pe.N_Dtheta] = cocir_edgelist[:,  ii_theta + i_theta]
                        cocir_edgelist[0:2, ii_theta + i_theta +  self.pe.N_r * self.pe.N_Dtheta] = self.pe.N_X - cocir_edgelist[0,  ii_theta + i_theta], self.pe.N_Y - cocir_edgelist[1, ii_theta + i_theta]
                cocir_edgelist[:, -1] = [self.pe.N_X /2, self.pe.N_Y /2, 0, edge_scale, cocir_edgelist[4,:].max() *1.2, 0.]
                return self.show_edges(cocir_edgelist, fig=fig, a=a, image=None, v_min=0., v_max=v_hist_noscale.max(), color=color)
            except Exception as e:
                self.log.error(' failed to generate cocir_geisler plot, %s', traceback.print_tb(sys.exc_info()[2]))
                return e, None # HACK to retrun something instead of None

        elif display=='cohist_scale':
            try:
                if fig==None:
                    fig = plt.figure(figsize=(self.pe.figsize_cohist, self.pe.figsize_cohist))
                    if a==None:
                        a = fig.add_subplot(111)
                a.bar(self.edges_loglevel[:-1], v_hist.sum(axis=(0, 1, 2)))
                plt.setp(a, yticks=[])
                a.set_xlabel('log2 of scale ratio')
                a.set_ylabel('probability')
                return fig, a
            except:
                self.log.error(' failed to generate cohist_scale, %s', e)
                return e, None # HACK to retrun something instead of None


        elif display=='chevrons':
            assert(symmetry==True)
            v_hist_angle = v_hist.sum(axis=0).sum(axis=-1) # -d-,phi,  theta, -scale-
            # some useful normalizations
            if not(prior==None):
                # this allows to show conditional probability by dividing by an arbitrary (prior) distribution
                prior_angle = prior.sum(axis=0).sum(axis=-1) # -d-, phi, theta, -scale-
                prior_angle /= prior_angle.sum()
                v_hist_angle /= prior_angle

            v_hist_angle /= v_hist_angle.mean()

            if dolog:
                v_hist_angle = np.log2(v_hist_angle)
            if v_max==None: v_max=v_hist_angle.max()
            if v_min==None: v_min=v_hist_angle.min()
#             v_hist_angle /= v_max

            # Computes the centers of the bins
            if half:
                v_phi, v_theta = self.edges_phi[(self.pe.N_phi/2-1):-1] + np.pi/self.pe.N_phi/2, self.edges_theta[(self.pe.N_Dtheta/2-1):-1] + np.pi/self.pe.N_Dtheta/2
                i_phi_shift, i_theta_shift = self.pe.N_phi/2+1, self.pe.N_Dtheta/2-1
            else:
                v_phi, v_theta = self.edges_phi - np.pi/self.pe.N_phi/2, self.edges_theta - np.pi/self.pe.N_Dtheta/2
                i_phi_shift, i_theta_shift = 2, -1
            s_phi, s_theta = len(v_phi), len(v_theta)
            #print 'DEBUG: s_phi, s_theta, self.pe.N_phi, self.pe.N_Dtheta', s_phi, s_theta, self.pe.N_phi, self.pe.N_Dtheta
            rad_X, rad_Y = 1.* self.pe.N_X/s_theta, 1.*self.pe.N_Y/s_phi
            rad = min(rad_X, rad_Y) / 2.619
            if radius==None: radius = np.ones((self.pe.N_phi, self.pe.N_Dtheta))

            if fig==None:
                fig = plt.figure(figsize=(self.pe.figsize_cohist, self.pe.figsize_cohist))
                if a==None:
                    border = 0.005
                    a = fig.add_axes((border, border, 1.-2*border, 1.-2*border), axisbg='w')
                    a.axis(c='b', lw=0)

            # make circles around each couple of edges
            import matplotlib.patches as patches
            from matplotlib.collections import PatchCollection
            import matplotlib.cm as cm
            import matplotlib.colors as mplcolors
            mypatches, colors = [], []
            fc=cm.gray(.6, alpha=.2) #cm.hsv(2./3., alpha=weight)#))
            angle_edgelist = np.zeros((6,  s_phi * s_theta * 2 ))
            for i_phi, phi in enumerate(v_phi):
                for i_theta, theta in enumerate(v_theta):
                    value = v_hist_angle[(s_phi - i_phi - i_phi_shift) % self.pe.N_phi, (i_theta + i_theta_shift) % self.pe.N_Dtheta]
                    score = radius[(s_phi - i_phi - i_phi_shift) % self.pe.N_phi, (i_theta + i_theta_shift) % self.pe.N_Dtheta]
                    circ = patches.Circle((rad_Y * (i_phi + .5) + .5,
                                       self.pe.N_X - rad_X * (s_theta - i_theta - .5) + .5),
                                       rad)#self.pe.line_width_chevrons/2)
                    mypatches.append(circ)
                    colors.append(value*score)

                    # first edge
#                    print i_phi, i_theta,  s_phi, s_theta, v_hist_angle.shape
                    angle_edgelist[0, i_phi * s_theta + i_theta] = self.pe.N_X - rad_X * (s_theta - i_theta - .5)
                    angle_edgelist[1, i_phi * s_theta + i_theta] = rad_Y * (i_phi + .5) - rad * 1.
                    angle_edgelist[2, i_phi * s_theta + i_theta] = phi + theta/2
                    angle_edgelist[3, i_phi * s_theta + i_theta] = self.pe.edge_scale_chevrons
                    angle_edgelist[4, i_phi * s_theta + i_theta] = 1.
                    # second edge
                    angle_edgelist[0, i_phi * s_theta + i_theta + s_phi * s_theta] = self.pe.N_X - rad_X * (s_theta - i_theta - .5)
                    angle_edgelist[1, i_phi * s_theta + i_theta + s_phi * s_theta] = rad_Y * (i_phi + .5) +  rad * 1.
                    angle_edgelist[2, i_phi * s_theta + i_theta + s_phi * s_theta] = phi - theta/2
                    angle_edgelist[3, i_phi * s_theta + i_theta + s_phi * s_theta] = self.pe.edge_scale_chevrons
                    angle_edgelist[4, i_phi * s_theta + i_theta + s_phi * s_theta] = 1.

            from matplotlib.colors import Normalize

            # see also http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib/7741317#7741317
            class MidpointNormalize(Normalize):
                def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False, gamma=1.):
                    self.midpoint = midpoint
                    self.gamma = gamma
                    Normalize.__init__(self, vmin, vmax, clip)

                def __call__(self, value, clip=None):
                    # I'm ignoring masked values and all kinds of edge cases to make a
                    # simple example...
                    x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                    z = np.ma.masked_array(np.interp(value, x, y))
                    pm = np.sign(z-.5)
                    z = pm*(np.absolute(z-.5)**(1/self.gamma))+.5
                    return z

#             p = PatchCollection(mypatches, norm=MidpointNormalize(midpoint=0, vmin=v_min, vmax=v_max), cmap=matplotlib.cm.RdBu_r, alpha=0.8)
            p = PatchCollection(mypatches, norm=MidpointNormalize(midpoint=0, vmin=v_min, vmax=v_max), cmap=cm.coolwarm, lw=0., alpha=1.0)

            p.set_array(np.array(colors))
            if dolog:
                p.set_clim([v_min, v_max])
            else:
                p.set_clim([v_min, v_max])
            a.add_collection(p)

#            print rad/s_theta, rad/s_phi
            fig, a = self.show_edges(angle_edgelist, fig=fig, a=a, image=None, color='black')
            a.axis([0, self.pe.N_Y+1, self.pe.N_X+1, 0])

            if colorbar:
                cbar = plt.colorbar(ax=a, mappable=p, shrink=0.6)
                if dolog:
                    if cbar_label: cbar.set_label('probability ratio')
                    ticks_cbar = 2**(np.floor(np.linspace(v_min, v_max, 3)))
                    cbar.set_ticks(np.log2(ticks_cbar))
                    cbar.set_ticklabels([r'$%0.1f$' % r for r in ticks_cbar])
#                     cbar.set_ticklabels([r'$2^{%d}$' % r for r in np.floor(np.log2(ticks_cbar))])
                else:
                    if cbar_label: cbar.set_label('probability ratio')
                    cbar.set_ticklabels(np.linspace(v_min, v_max, 5))#, base=2))
                cbar.update_ticks()

            if not(labels==False):
                if not(xticks=='left'): a.set_xlabel(r'azimuth difference $\psi$')
                if not(xticks=='bottom'): a.set_ylabel(r'orientation difference $\theta$')
            if not(xticks==False):
                eps = 0.5 # HACK to center grid. dunnon what's happening here
                if half:
                    plt.setp(a, xticks=[(1./self.pe.N_phi/1.25)*self.pe.N_X, (1. - 1./self.pe.N_phi/1.25)*self.pe.N_X])
                    if not(xticks=='left'):
                        plt.setp(a, xticklabels=[r'$0$', r'$\frac{\pi}{2}$'])
                    else:
                        plt.setp(a, xticklabels=[r'', r''])
                else:
                    plt.setp(a, xticks=[(1./(self.pe.N_phi+1)/2)*self.pe.N_X+eps, .5*self.pe.N_X+eps, (1. - 1./(self.pe.N_phi+1)/2)*self.pe.N_X+eps])
                    if not(xticks=='left'):
                        plt.setp(a, xticklabels=[r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$'])
                    else:
                        plt.setp(a, xticklabels=[r'', r''])
                if half:
                    plt.setp(a, yticks=[(1./self.pe.N_Dtheta)*self.pe.N_Y, (1. - 1./(self.pe.N_Dtheta+.45))*self.pe.N_Y])
                    if not(xticks=='bottom'):
                        plt.setp(a, yticklabels=[r'$0$', r'$\frac{\pi}{2}$'])
                    else:
                        plt.setp(a, yticklabels=[r'', r''])
                else:
                    plt.setp(a, yticks=[1./(self.pe.N_Dtheta+1)/2*self.pe.N_X+eps, .5*self.pe.N_Y+eps, (1. - 1./(self.pe.N_Dtheta+1)/2)*self.pe.N_Y+eps])
                    if not(xticks=='bottom'):
                        plt.setp(a, yticklabels=[r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$'])
                    else:
                        plt.setp(a, yticklabels=['', '', ''])
                plt.grid('off')
            plt.draw()

            return fig, a
        else:
            return v_hist


    def process(self, exp, name_database='serre07_distractors', note='', noise=0.):
        """
        The pipeline to go from one database to a list of edge lists

        ``note`` designs a string that modified the histogram (such as changing the number of bins)

        """

        self.log.info(' > computing edges for experiment %s with database %s ', exp, name_database)
        #: 1 - Creating an image list
        locked = False
        matname = os.path.join(self.pe.edgematpath, exp + '_' + name_database)
        #while os.path.isfile(matname + '_images_lock'):
        imagelist = self.get_imagelist(exp, name_database=name_database)
        locked = (imagelist=='locked')
#         print 'DEBUG: theta used in this experiment: ', self.theta*180/np.pi
        # 2- Doing the edge extraction for each image in this list
        if not(locked):
            try:
                edgeslist = np.load(matname + '_edges.npy')
            except Exception as e:
                self.log.info(' >> There is no edgeslist: %s ', e)
#                 self.log.info('>> Doing the edge extraction')
                time.sleep(1.*np.random.rand())
                edgeslist = self.full_run(exp, name_database, imagelist, noise=noise)
                if edgeslist == 'locked':
                    self.log.info('>> Edge extraction %s is locked', matname)
                    locked = True
                else:
                    np.save(matname + '_edges.npy', edgeslist)
        else:
            return 'locked imagelist', 'not done', 'not done'

        # 3- Doing the independence check for this set
        if not(locked):
            txtname = os.path.join(self.pe.figpath, exp + '_dependence_' + name_database + note + '.txt')
            if not(os.path.isfile(txtname)) and not(os.path.isfile(txtname + '_lock')):
                open(txtname + '_lock', 'w').close() # touching
                self.log.info(' >> Doing check_independence on %s ', txtname)
                out = self.check_independence(self.cohistedges(edgeslist, symmetry=False, display=None), name_database, exp)
                f = open(txtname, 'w')
                f.write(out)
                f.close()
                print(out)
                try:
                    os.remove(txtname + '_lock')
                except Exception as e:
                    self.log.error('Failed to remove lock file %s_lock, error : %s ', txtname, e)

        # 4- Doing the edge figures to check the edge extraction process
        edgedir = os.path.join(self.pe.edgefigpath, exp + '_' + name_database)
        if not(os.path.isdir(edgedir)): os.mkdir(edgedir)

        if not(locked):
            N_image = edgeslist.shape[2]
            for index in np.random.permutation(np.arange(len(imagelist))):
                filename, croparea = imagelist[index]

                figname = os.path.join(edgedir, filename.replace('.png', '') + str(croparea) + '.png')
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    try:
                        open(figname + '_lock', 'w').close()
                        self.log.info('> redoing figure %s ', figname)
                        image, filename_, croparea_ = self.patch(name_database=name_database, filename=filename, croparea=croparea)
                        if noise >0.: image += noise*image[:].std()*self.texture(filename=filename, croparea=croparea)
                        if self.pe.do_whitening: image = self.whitening(image)
                        fig, a = self.show_edges(edgeslist[:, :, index], image=image*1.)
                        plt.savefig(figname)
                        plt.close('all')
                        try:
                            os.remove(figname + '_lock')
                        except Exception as e:
                            self.log.info('Failed to remove lock file %s_lock , error : %s ', figname , e)
                    except Exception as e:
                        self.log.info('Failed to make edge image  %s, error : %s ', figname , traceback.print_tb(sys.exc_info()[2]))

                figname = os.path.join(edgedir, filename.replace('.png', '') + str(croparea) + '_reconstruct.png')
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    try:
                        open(figname + '_lock', 'w').close()
                        self.log.info('> reconstructing figure %s ', figname)
                        image_ = self.reconstruct(edgeslist[:, :, index])
#                         if self.pe.do_whitening: image_ = self.dewhitening(image_)
                        fig, a = self.show_edges(edgeslist[:, :, index], image=image_*1.)
                        plt.savefig(figname)
                        plt.close('all')
                        try:
                            os.remove(figname + '_lock')
                        except Exception as e:
                            self.log.error('Failed to remove lock file %s_lock, error : %s ', figname, traceback.print_tb(sys.exc_info()[2]))
                    except Exception as e:
                        self.log.error('Failed to make reconstruct image  %s , error : %s  ', figname, traceback.print_tb(sys.exc_info()[2]))

            # 5- Computing RMSE to check the edge extraction process
            try:
                RMSE = np.load(matname + '_RMSE.npy')
            except Exception as e:
                self.log.info(' >> There is no RMSE: %s ', e)
                try:
                    RMSE = self.full_RMSE(exp, name_database, imagelist)
                    if RMSE is 'locked':
                        self.log.info('>> RMSE extraction %s is locked', matname)
                        locked = True
                    else:
                        np.save(matname + '_RMSE.npy', RMSE)
                except Exception as e:
                    self.log.error('Failed to compute RMSE %s , error : %s ', matname + '_RMSE.npy', e)

            try:
                self.log.info('>>> For the class %s, in experiment %s RMSE = %f ', name_database, exp, (RMSE[:, -1]/RMSE[:, 0]).mean())
            except Exception as e:
                self.log.error('Failed to display RMSE %s ', e)
            # 6- Plotting the histogram
            try:
#            figname = os.path.join(self.pe.figpath, exp + '_proba-scale_' + name_database + note + self.pe.ext)
#            if not(os.path.isfile(figname)):
#                fig, a = self.histedges_scale(edgeslist, display=True)
#                plt.savefig(figname)
#                plt.close('all')
#
                figname = os.path.join(self.pe.figpath, exp + '_proba-theta_' + name_database + note + self.pe.ext)
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    open(figname + '_lock', 'w').close()
                    fig, a = self.histedges_theta(edgeslist, display=True)
                    plt.savefig(figname)
                    plt.close('all')
                    os.remove(figname + '_lock')

                figname = os.path.join(self.pe.figpath, exp + '_proba-edgefield_colin_' + name_database + note + self.pe.ext)
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    open(figname + '_lock', 'w').close()
                    fig, a = self.cohistedges(edgeslist, symmetry=False, display='colin_geisler')
                    plt.savefig(figname)
                    plt.close('all')
                    os.remove(figname + '_lock')

                figname = os.path.join(self.pe.figpath, exp + '_proba-edgefield_cocir_' + name_database + note + self.pe.ext)
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    open(figname + '_lock', 'w').close()
                    fig, a = self.cohistedges(edgeslist, symmetry=False, display='cocir_geisler')
                    plt.savefig(figname)
                    plt.close('all')
                    os.remove(figname + '_lock')

                figname = os.path.join(self.pe.figpath, exp + '_proba-edgefield_chevrons_' + name_database + note + self.pe.ext)
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    open(figname + '_lock', 'w').close()
                    fig, a = self.cohistedges(edgeslist, display='chevrons')
                    plt.savefig(figname)
                    plt.close('all')
                    os.remove(figname + '_lock')

                if 'targets' in name_database or 'laboratory' in name_database:
                    figname = os.path.join(self.pe.figpath, exp + '_proba-edgefield_chevrons_priordistractors_' + name_database + '_' + note + self.pe.ext)
                    if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                        open(figname + '_lock', 'w').close()
                        imagelist_prior = self.get_imagelist(exp, name_database=name_database.replace('targets', 'distractors'))
                        edgeslist_prior = self.full_run(exp, name_database.replace('targets', 'distractors'), imagelist_prior, noise=noise)
                        v_hist_prior = self.cohistedges(edgeslist_prior, display=None)
                        fig, a = self.cohistedges(edgeslist, display='chevrons', prior=v_hist_prior)
                        plt.savefig(figname)
                        plt.close('all')
                        os.remove(figname + '_lock')
            except Exception as e:
                self.log.error('Failed to create figures, error : %s ', e)

            return imagelist, edgeslist, RMSE
        else:
            return 'locked', 'locked edgeslist', ' locked RMSE '

    # some helper funtion to compare the databases
    def KL(self, v_hist, v_hist_obs):
        if v_hist.sum()==0 or v_hist_obs.sum()==0: self.log.error('>X>X>X KL function:  problem with null histograms! <X<X<X<')
        elif True:
            v_hist /= v_hist.sum()
            v_hist_obs /= v_hist_obs.sum()
            # taking advantage of log(True) = 0 and canceling out null bins in v_hist_obs
            kl = np.sum(v_hist.ravel()*(np.log(v_hist.ravel()+(v_hist == 0).ravel())
                                        - np.log(v_hist_obs.ravel()+(v_hist_obs == 0).ravel())))
            if kl == np.nan: print ( v_hist.sum(), v_hist_obs.sum() )
            return kl
        else:
            from scipy.stats import entropy
            return entropy(v_hist_obs, v_hist, base=2)

    def check_independence(self, v_hist, name_database, exp, labels=['d', 'phi', 'theta', 'scale']):
        v_hist /= v_hist.sum()
        fullset = [0, 1, 2, 3]
#    from scipy.stats import entropy
#    print KL(v_hist, v_hist),  entropy(v_hist.ravel(), v_hist.ravel())
        flat = np.ones_like(v_hist)
        flat /= flat.sum()
        out = 'Checking dependence in ' + name_database + '_' + exp + '\n'
        out += '-'*60 + '\n'
        out += 'Entropy: ' + str(self.KL(v_hist, flat)) + '\n'
        out += '-'*60 + '\n'
        combinations = [[[0, 1, 2, 3]], # full dependence
                         [[1, 2, 3], [0]],
                         [[2, 3, 0], [1]],
                         [[3, 0, 1], [2]],
                         [[0, 1, 2], [3]],
                         [[1, 2], [3, 0]],
                         [[2, 3], [0, 1]],
                         [[3, 1], [0, 2]],
                         [[1, 2], [3], [0]],
                         [[2, 3], [0], [1]],
                         [[3, 0], [1], [2]],
                         [[0, 1], [2], [3]],
                         [[0], [1], [2], [3]], # full independence
                         ]
        out += '-'*60 + '\n'

        def marginal(v_hist, subset):
            """
            marginalize the distribution v_hist over the variables given in subset
            uses a recursive approach, marginaluzung over axis individually

            """
            if subset == []:
                return v_hist
            else:
                v_hist_ = v_hist.copy() # np.ones_like(v_hist)
                for axis_ in subset:
                    v_hist_ = np.expand_dims(v_hist_.mean(axis=axis_), axis=axis_)*np.ones_like(v_hist_)
                return v_hist_

        for combination in combinations:
            combination_str = str([[labels[k] for k in subset ] for subset in combination])
#        print combination_str
            combination_str.replace('[[', 'p(')
            combination_str.replace(']]', ')')
            combination_str.replace(', [', '.p(')
            combination_str.replace(')', ')')
            combination_str.strip("'")
#        print combination_str
            # computing marginalized distribution as an approximation
            v_hist_ = np.ones_like(v_hist)
            for subset in combination:
#            print subset, [k for k in fullset if k not in subset]
                v_hist_ *= marginal(v_hist, [k for k in fullset if k not in subset])
            v_hist_ /= v_hist_.sum()
            out += combination_str + ' KL= ' + '%.5f' % self.KL(v_hist, v_hist_) + ' ; ' + '%.3f' % (self.KL(v_hist, v_hist_)/self.KL(v_hist, flat)*100) + '\n'
        out += '-'*60 + '\n'
        return out

    def plot(self, mps, experiments, databases, labels, fig=None, ax=None,
            color=[1., 0., 0.], threshold=None, scale=False, ref=None,  revert=False):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib

        plt.rc('axes', linewidth=.25)
        plt.rc('axes', edgecolor='black')
        if fig==None:
            # parameters for plots
            fig_width_pt = 318.670  # Get this from LaTeX using \showthe\columnwidth
            inches_per_pt = 1.0/72.27               # Convert pt to inches
            fig_width = fig_width_pt*inches_per_pt  # width in inches

            fig = plt.figure(figsize=(fig_width, fig_width/1.618))
        # main axis
        if ax==None: ax = fig.add_subplot(111, axisbg='w')
        # axes.edgecolor      : black   # axes edge color
        if (threshold==None) and (ref==None):
            if revert:
                inset = fig.add_subplot(111, axisbg='w')
                # this is another inset axes over the main axes
                ax = fig.add_axes([0.48, 0.55, .4, .4], axisbg='w')
            else:
                ax = fig.add_subplot(111, axisbg='w')
                # this is another inset axes over the main axes
                inset = fig.add_axes([0.48, 0.55, .4, .4], axisbg='w')
            #CCycle = np.vstack((np.linspace(0, 1, len(experiments)), np.zeros(len(experiments)), np.zeros(len(experiments)))).T
            grad = np.linspace(0., 1., 2*len(experiments))
            grad[1::2] = grad[::2]
            CCycle = np.array(color)[np.newaxis, :] * grad[:, np.newaxis]
            ax.set_color_cycle(CCycle)
            inset.set_color_cycle(CCycle)
            l0_max, eev = 0., -len(experiments)/2
            for mp, experiment, name_database, label in zip(mps, experiments, databases, labels):
                try:
                    imagelist, edgeslist, RMSE = mp.process(exp=experiment, name_database=name_database)
                    # print(RMSE.shape, RMSE[:, 0])
                    N = RMSE.shape[1] #number of edges
                    l0_max = max(l0_max, N*np.log2(mp.oc)/mp.pe.N_X/mp.pe.N_Y)
                    if not(scale):
                        l0_axis = np.arange(N)
                    else:
                        l0_axis = np.linspace(0, N*np.log2(mp.oc)/mp.pe.N_X/mp.pe.N_Y, N)
                    errorevery_zoom = 1.4**(1.*eev/len(experiments))
                    try:
                        RMSE /= RMSE[:, 0][:, np.newaxis]
                        errorevery = np.max((int(RMSE.shape[1]/8*errorevery_zoom), 1))
                        ax.errorbar(l0_axis, RMSE.mean(axis=0),
                                    yerr=RMSE.std(axis=0), label=label, errorevery=errorevery)
                        ax.plot(l0_axis[::errorevery], RMSE.mean(axis=0)[::errorevery],
                                    linestyle='None', marker='o', ms=3)
                    except Exception as e:
                        print('Failed to plot RMSE in experiment %s with error : %s ' % (experiment, e) )
                    try:
                        coeff0 = edgeslist[4, 0, :].mean()
                        inset.errorbar(l0_axis, edgeslist[4, :, :].mean(axis=1)/coeff0,
                                    yerr=edgeslist[4, :, :].std(axis=1)/coeff0, label=label, errorevery=errorevery)
                        inset.plot(l0_axis[::errorevery], edgeslist[4, :, :].mean(axis=1)[::errorevery]/coeff0,
                                    linestyle='None',  marker='o', ms=3)
                    except Exception as e:
                        print('Failed to plot coeffs in experiment %s with error : %s ' % (experiment, e) )
                    eev += 1
                except Exception as e:
                    print('Failed to load data to plot experiment %s with error : %s ' % (experiment, e) )
            for a in [ax, inset]:
                #a.set_yscale("log")#, nonposx = 'clip')
                if not(scale):
                    a.set_xlim([-0.05*N, 1.05*N])
                else:
                    a.set_xlim([-0.05*l0_max, 1.05*l0_max])
                    a.ticklabel_format(axis='x', style='sci', scilimits=(0, 1))#, useOffset=False)
                #a.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                a.spines['left'].set_position('zero')#('outward', -10))
                a.spines['right'].set_visible(False)
                a.spines['bottom'].set_position('zero')#(('outward', -10))
                a.spines['top'].set_visible(False)
                #a.spines['left'].set_smart_bounds(True)
                a.spines['bottom'].set_smart_bounds(True)
                a.xaxis.set_ticks_position('bottom')
                a.yaxis.set_ticks_position('left')
                if not(scale):#False and a==ax:
                    a.set_xlabel(r'$\ell_0$-norm')
                else:
                    ax.set_xlabel(r'relative $\ell_0$ pseudo-norm (bits / pixel)')#relative $\ell_0$-norm')

                a.grid(b=False, which="both")

            ax.set_ylim(-.02, 1.02)
            ax.set_ylabel(r'Squared error')
            inset.set_ylabel(r'Coefficient')
            if revert:
                ax.legend(loc='best', frameon=False)#, bbox_to_anchor = (0.5, 0.5))
            else:
                inset.legend(loc='best', frameon=False, bbox_to_anchor = (0.4, 0.4))
            plt.locator_params(tight=False, nbins=4)
            plt.tight_layout()
            return fig, ax, inset
        elif (threshold==None):
            if ax==None: ax = fig.add_axes([0.15, 0.25, .75, .75], axisbg='w')
            ind, l0, l0_std = 0, [], []
            from lmfit.models import ExpressionModel
            mod = ExpressionModel('1 - (1- eps_inf) * ( 1 - rho**x)')
            for mp, experiment, name_database, label in zip(mps, experiments, databases, labels):
                try:
                    imagelist, edgeslist, RMSE = mp.process(exp=experiment, name_database=name_database)
                    RMSE /= RMSE[:, 0][:, np.newaxis]
                    N = RMSE.shape[1] #number of edges
                    if RMSE.min()>threshold: print('the threshold is never reached for', experiment, name_database)
                    try:
                        l0_results = np.zeros(N)
                        for i_image in range(RMSE.shape[0]):
                            mod.def_vals = {'eps_inf':.1, 'rho':.99}
                            out  = mod.fit(RMSE[i_image, :], x=np.arange(N))
                            eps_inf = out.params.get('eps_inf').value
                            rho =  out.params.get('rho').value
                            #print rho, eps_inf, np.log((threshold-eps_inf)/(1-eps_inf))/np.log(rho)

                            l0_results[i_image] = np.log((threshold-eps_inf)/(1-eps_inf))/np.log(rho)
                    except:
                        l0_results = np.argmax(RMSE<threshold, axis=1)*1.
                    if (scale):
                        l0_results *= np.log2(mp.oc)/mp.pe.N_X/mp.pe.N_Y
                    l0.append(l0_results.mean())
                    l0_std.append(l0_results.std())
                    ind += 1
                except Exception as e:
                    print('Failed to plot experiment %s with error : %s ' % (experiment, e) )


#  subplots_adjust(left=None, bottom=None, right=None, top=None,
#                  wspace=None, hspace=None)
#
#The parameter meanings (and suggested defaults) are::
#
#  left  = 0.125  # the left side of the subplots of the figure
#  right = 0.9    # the right side of the subplots of the figure
#  bottom = 0.1   # the bottom of the subplots of the figure
#  top = 0.9      # the top of the subplots of the figure
#  wspace = 0.2   # the amount of width reserved for blank space between subplots
#  hspace = 0.2   # the amount of height reserved for white space between subplots
            fig.subplots_adjust(wspace=0.1, hspace=0.1,
                                left=0.2, right=0.9,
                                top=0.9,    bottom=0.175)

            width = .8
            ax.bar(np.arange(ind), l0)#, yerr=l0_std)
            ax.set_xlim([-width/4, ind+.0*width])

            if not(scale):#False and a==ax:
                ax.set_ylabel(r'$\ell_0$ pseudo-norm')
            else:
#             ax.set_ylabel(r'relative $\ell_0$ pseudo-norm')# (bits / pixel)')#relative $\ell_0$-norm')
                ax.set_ylabel(r'rel. $\ell_0$ norm')# (bits / pixel)')#relative $\ell_0$-norm')

            ax.set_xticks(np.arange(ind)+.5*width)
            ax.set_xticklabels(labels)

            plt.tight_layout()
#         fig.set_tight_layout(True)

            return fig, ax, ax

        elif (ref==None):
            absSE, absSE_std = [], []

            for mp, experiment, name_database, label in zip(mps, experiments, databases, labels):
                imagelist, edgeslist, RMSE = mp.process(exp=experiment, name_database=name_database)
                RMSE /= RMSE[:, 0][:, np.newaxis]
                N = RMSE.shape[1] #number of edges
                l0 = np.log2(mp.oc)/mp.pe.N_X/mp.pe.N_Y
                absSE.append(RMSE.mean())
                absSE_std.append(RMSE.std(axis=0).mean())
            fig.subplots_adjust(wspace=0.1, hspace=0.1,
                                left=0.2, right=0.9,
                                top=0.9,    bottom=0.175)

            ind = len(absSE)
            width = .8
            ax.bar(np.arange(ind), absSE, yerr=absSE_std)
            ax.set_xlim([-width/4, ind+.0*width])

            if not(scale):#False and a==ax:
                ax.set_ylabel(r'SE')
            else:
#             ax.set_ylabel(r'absative $\ell_0$ pseudo-norm')# (bits / pixel)')#absative $\ell_0$-norm')
                ax.set_ylabel(r'abs. SE')# (bits / pixel)')#absative $\ell_0$-norm')

            ax.set_xticks(np.arange(ind)+.5*width)
            ax.set_xticklabels(labels)

            plt.tight_layout()
#         fig.set_tight_layout(True)

            return fig, ax, ax

        else: # fourth type: we have a reference and a threshold
            try:
                relL0, relL0_std = [], []
                # computes for the reference
                imagelist_ref, edgeslist_ref, RMSE_ref = mps[ref].process(exp=experiments[ref], name_database=databases[ref])
                RMSE_ref /= RMSE_ref[:, 0][:, np.newaxis] # normalize RMSE
                L0_ref =  np.argmax(RMSE_ref<threshold, axis=1)*1. +1
                if scale: L0_ref *= np.log2(mps[ref].oc)/mps[ref].N_X/mps[ref].N_Y
#             print("ref-thr - L0_ref=", L0_ref)

                for mp, experiment, name_database, label in zip(mps, experiments, databases, labels):
                    imagelist, edgeslist, RMSE = mp.process(exp=experiment, name_database=name_database)
                    RMSE /= RMSE[:, 0][:, np.newaxis] # normalize RMSE
                    N = RMSE.shape[1] #number of edges
                    L0 =  np.argmax(RMSE<threshold, axis=1)*1.
                    if RMSE.min()>threshold: print('the threshold is never reached for', experiment, name_database)
                    if scale: L0 *= np.log2(mp.oc)/mp.pe.N_X/mp.pe.N_Y
                    relL0.append((L0/L0_ref).mean())
                    relL0_std.append((L0/L0_ref).std())

                fig.subplots_adjust(wspace=0.1, hspace=0.1,
                                    left=0.2, right=0.9,
                                    top=0.9,    bottom=0.175)

                ind = len(relL0)
                width = .8
#             print("ref-thr - relL0=", relL0)
                rects = ax.bar(np.arange(ind), relL0, yerr=relL0_std, alpha=.8, error_kw={'ecolor':'k'})
                rects[ref].set_color('w')
                rects[ref].set_edgecolor('k')
                ax.set_xlim([-width/4, ind+.0*width])

                ax.set_ylabel(r'relative coding cost wrt default')# (bits / pixel)')#relative $\ell_0$-norm')

                ax.set_xticks(np.arange(ind)+.5*width)
                ax.set_xticklabels(labels)

                ax.grid(b=False, which="both")
#         plt.tight_layout()
#         fig.set_tight_layout(True)

                return fig, ax, ax

            except Exception as e:
                print('Failed to analyze experiment %s with error : %s ' % (experiment, e) )

    def golden_pyramid(self, z):
        """
        TODO : put in LogGabor

        """

        phi = (np.sqrt(5) +1.)/2. # golden ratio
        opts= {'vmin':0., 'vmax':1., 'interpolation':'nearest', 'origin':'upper'}
        fig_width = 13
        fig = plt.figure(figsize=(fig_width, fig_width/phi))
        xmin, ymin, size = 0, 0, 1.
        for i_sf_0, sf_0_ in enumerate(self.sf_0):
            a = fig.add_axes((xmin/phi, ymin, size/phi, size), axisbg='w')
            a.axis(c='b', lw=0)
            plt.setp(a, xticks=[])
            plt.setp(a, yticks=[])
            im_RGB = np.zeros((self.pe.N_X, self.pe.N_Y, 3))
            for i_theta, theta_ in enumerate(self.theta):
                im_abs = np.absolute(z[:, :, i_theta, i_sf_0])
                RGB = np.array([.5*np.sin(2*theta_ + 2*i*np.pi/3)+.5 for i in range(3)])
                im_RGB += im_abs[:,:, np.newaxis] * RGB[np.newaxis, np.newaxis, :]

            im_RGB /= im_RGB.max()
            a.imshow(im_RGB, **opts)
            #a.grid(False)
            a.grid(b=False, which="both")
            i_orientation = np.mod(i_sf_0, 4)
            if i_orientation==0:
                xmin += size
                ymin += size/phi**2
            elif i_orientation==1:
                xmin += size/phi**2
                ymin += -size/phi
            elif i_orientation==2:
                xmin += -size/phi
            elif i_orientation==3:
                ymin += size
            size /= phi
        return fig


class SparseEdgesWithDipole(SparseEdges):
    def __init__(self, pe):
        """
        Extends the SparseEdges class by includiong a dipole, see

        http://invibe.net/Publications/Perrinet15eusipco

        """
        SparseEdges.__init__(self, pe=pe)
#         self.init()
        self.init_logging(name='SparseEdgesWithDipole')

        self.pe.eta_SO =  0.
        self.pe.dip_w =  0.2
        self.pe.dip_B_psi =  0.1
        self.pe.dip_B_theta =  1.
        self.pe.dip_scale =  1.5
        self.pe.dip_epsilon =  .5

    def run_mp(self, image, verbose=False):
        """
        runs the MatchingPursuit algorithm on image

        """
        edges = np.zeros((6, self.pe.N))
        image_ = image.copy()
#         residual = image.copy()
#         RMSE = np.ones(self.pe.N)
        if self.pe.do_whitening: image_ = self.whitening(image_)
        C = self.init_C(image_)
        logD = np.zeros((self.pe.N_X, self.pe.N_Y, self.pe.n_theta, self.n_levels), dtype=np.complex)
        if verbose:
            import pyprind
            my_prbar = pyprind.ProgPercent(self.pe.N)   # 1) initialization with number of iterations
        for i_edge in range(self.pe.N):
#             RMSE[i_edge] = np.sum((residual - image_)**2)
            # MATCHING
            ind_edge_star = self.argmax(C * np.exp( self.pe.eta_SO * logD))
            if not self.pe.MP_rho is None:
                if i_edge==0: C_Max = np.absolute(C[ind_edge_star])
                coeff = self.pe.MP_alpha * (self.pe.MP_rho ** i_edge) *C_Max
                # recording
                if verbose: print('Edge', i_edge, '/', self.pe.N, ' - Max activity (quant mode) : ', np.absolute(C[ind_edge_star]), ', coeff/alpha=', coeff/self.pe.MP_alpha , ' phase= ', np.angle(C[ind_edge_star], deg=True), ' deg,  @ ', ind_edge_star)
            else:
                coeff = self.pe.MP_alpha * np.absolute(C[ind_edge_star])
                # recording
                if verbose: print('Edge', i_edge, '/', self.pe.N, ' - Max activity  : ', np.absolute(C[ind_edge_star]), ' phase= ', np.angle(C[ind_edge_star], deg=True), ' deg,  @ ', ind_edge_star)
            if verbose: my_prbar.update()
            edges[:, i_edge] = np.array([ind_edge_star[0]*1., ind_edge_star[1]*1.,
                                         self.theta[ind_edge_star[2]],
                                         self.sf_0[ind_edge_star[3]],
                                         coeff, np.angle(C[ind_edge_star])])
            # PURSUIT
            if self.pe.eta_SO>0.: logD+= np.absolute(C[ind_edge_star]) * self.dipole(edges[:, i_edge])
            C = self.backprop(C, ind_edge_star)
        return edges, C


    def dipole(self, edge):

        y, x, theta_edge, sf_0, C, phase = edge # HACK
        theta_edge = np.pi/2 - theta_edge

        D = np.ones((self.pe.N_X, self.pe.N_Y, self.pe.n_theta, self.n_levels))
        distance = np.sqrt(((1.*self.X-x)**2+(1.*self.Y-y)**2)/(self.pe.N_X**2+self.pe.N_Y**2))/self.pe.dip_w
        neighborhood = np.exp(-distance**2)
        for i_sf_0, sf_0_ in enumerate(self.sf_0):
            for i_theta, theta_layer in enumerate(self.theta):
                theta_layer = np.pi/2 - theta_layer # HACK - to correct in +LogGabor
                theta_layer = ((theta_layer + np.pi/2 - np.pi/self.pe.n_theta/2)  % (np.pi) ) - np.pi/2  + np.pi/self.pe.n_theta/2
                theta = theta_layer - theta_edge # angle between edge's orientation and the layer's one
                psi = np.arctan2(self.Y-y, self.X-x) - theta_edge -np.pi/2 - theta/2 #- np.pi/4
                d = (1-self.pe.dip_epsilon)*distance + self.pe.dip_epsilon
                D[:, :, i_theta, i_sf_0] = np.exp((np.cos(2*psi)-1.)/(self.pe.dip_B_psi**2 * d))
                D[:, :, i_theta, i_sf_0] *= np.exp((np.cos(2*theta)-1.)/(self.pe.dip_B_theta**2 * d))
            D[:, :, :, i_sf_0] *= neighborhood[:, :, np.newaxis] * np.exp(-np.abs( np.log2(self.sf_0[i_sf_0] / sf_0)) / self.pe.dip_scale)
        #print np.exp(-np.abs( np.log2(self.sf_0 / sf_0)) / self.pe.dip_scale)
        D -= D.mean()
        D /= np.abs(D).max()

        return np.log2(1.+D)



class EdgeFactory(SparseEdges):
    """
    EdgeFactory

    A class which classifies images based on histograms of their statistics.

    We tested first a simplistic classifier ``compare`` and then use the ``SVM``
    classifier from sklearn.

    The pipeline is to

    * extract edges from all images,
    * create the representation (histogram) for each class,
    * fit the data,
    * classify and return the f1-score.

    """
    def svm(self, exp, opt_notSVM='', opt_SVM='', databases=['serre07_distractors', 'serre07_targets'],
            edgeslists = [None, None],
            feature='full', kernel = 'precomputed', KL_type='JSD', noise=0.):

        import time
        time.sleep(.1*np.random.rand())

        # DEFINING FILENAMES
        # put here thing that do change the histogram
        #opt_notSVM = opt_notSVM # (this should be normally passed to exp, as in 'classifier_noise')
        # and here things that do not change the (individual) histogramsi but
        # rather the SVM classification:
        if (self.pe.svm_log): opt_SVM += '_log'
#         if not(self.pe.svm_log): opt_SVM += 'nolog_'
        if self.pe.svm_norm: opt_SVM += '_norm'
        name_databases = ''
        for database in databases: name_databases += database + '_'
        txtname = os.path.join(self.pe.figpath, exp + '_SVM_' + name_databases + feature + opt_notSVM + opt_SVM +'.txt')
        matname_score = txtname.replace(self.pe.figpath, self.pe.matpath).replace('.txt', '.npy')
        # gathering all data: first the raw complete data, then the X, y
        # vectors; the final result is the matname_score file containg the
        # classification results

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
        for i_database, (name_database, edgeslist) in enumerate(zip(databases, edgeslists)):
            if edgeslist is None:
                imagelist, edgeslist, RMSE = self.process(exp, note=opt_notSVM, name_database=name_database, noise=noise)
            else:
                imagelist = 'ok'

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
                # Download the data, if not already on disk and load it as numpy arrays
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
                                imagelist, edgeslist, RMSE = self.process(exp, note=opt_notSVM, name_database=name_database, noise=noise)
                            else:
                                imagelist = 'ok'
                            try:#if not(type(imagelist) == 'str'):
                                t0 = time.time()
                                hists = []
                                for i_image in range(edgeslist.shape[2]):
                                    # TODO : use as features each edge co-occurence?
                                    # TODO : make full minus chevrons
                                    if feature_ == 'full':
                                        # using the full histogram
                                        v_hist = self.cohistedges(edgeslist[:, :, i_image][..., np.newaxis], display=None)
                                    elif feature_ == 'chevron':
                                        #  or just the chevron map
                                        v_hist = self.cohistedges(edgeslist[:, :, i_image][..., np.newaxis], display=None)
                                        # marginalize over distances and scales
                                        v_hist = v_hist.sum(axis=3).sum(axis=0)
                                    elif feature_ == 'first':
                                        # control with first-order
                                        v_hist, v_theta_edges_ = self.histedges_theta(edgeslist[:, :, i_image][..., np.newaxis], display=False)
                                    elif feature_ == 'first_rot':
                                        edgeslist[2, :, i_image] += np.random.rand() * np.pi
                                        # control with first-order
                                        v_hist, v_theta_edges_ = self.histedges_theta(edgeslist[:, :, i_image][..., np.newaxis], display=False)
                                    else:
                                        self.log.error('problem here, you asked for a non-existant feature', feature_)
                                        break
                                    # normalize histogram
                                    v_hist /= v_hist.sum()
                                    hists.append(v_hist.ravel())
                                hists = np.array(hists)
                                np.save(matname_hist, hists)
                                self.log.info("Histogram done in %0.3fs", (time.time() - t0))
                            except Exception as e:
                                self.log.error(' XX The process computing edges in %s is locked ', name_database)
                                self.log.error(' Raised exection %s  ', e)
                            try:
                                os.remove(matname_hist + '_lock')
                            except:
                                self.log.error(' xxx when trying to remove it, I found no lock file named %s_lock', matname_hist)

            # gather data
            locked = False
            X_, y_ = {}, []
            for feature_ in features:
                X_[feature_] = []
                for i_database, name_database in enumerate(databases):
                    matname_hist = os.path.join(self.pe.matpath, exp + '_SVM-hist_' + name_database + '_' + feature_ + opt_notSVM + '.npy')
                    try:
                        hists = np.load(matname_hist)
                        for i_image in range(hists.shape[0]):
                            X_[feature_].append(hists[i_image, :])
                    except Exception as e:
                        self.log.warn(' >> Missing histogram, skipping SVM : %s ', e)
                        locked = True
                        return None
            # TODO simplify the following
            for i_database, name_database in enumerate(databases):
                matname_hist = os.path.join(self.pe.matpath, exp + '_SVM-hist_' + name_database + '_' + feature_ + opt_notSVM + '.npy')
                try:
                    hists = np.load(matname_hist)
                    for i_image in range(hists.shape[0]):
                        y_.append(i_database)
                except Exception as e:
                    self.log.warn(' >> Missing histogram, skipping SVM : %s ', e)
                    locked = True
                    return None

            # starting SVM
            X = {}
            for feature_ in features:
                X[feature_] = np.array(X_[feature_])
            y = np.array(y_)

            # do the classification
            fone_score = np.zeros(self.pe.N_svm_cv)
            tested_indices, is_target, predicted_target = [], [], []
#             import hashlib
#             random_state = int("0x" +  hashlib.sha224(matname_score).hexdigest(), 0)*1.
#             random_state = int(self.pe.seed + random_state % 4294967295)
            t0_cv = time.time()
            for i_cv in range(self.pe.N_svm_cv):
                ###############################################################################
                # 1- Split into a training set and a test set using a stratified k fold
                from sklearn.cross_validation import ShuffleSplit
                rs = ShuffleSplit(y.shape[0], n_iter=1, test_size=self.pe.svm_test_size, random_state=i_cv)#random_state + i_cv)
                # split into a training and testing set
                for index_train, index_test in rs: pass
                X_train, X_test, y_train, y_test = {}, {}, [], []
                for feature_ in features:
                    X_train[feature_], X_test[feature_] = X[feature_][index_train, :], X[feature_][index_test, :]
                y_train, y_test =  y[index_train], y[index_test]
                n_train, n_test = len(y_train), len(y_test)
                is_target.append(y_test)
                tested_indices.append(index_test)

                # 2- normalization
                if self.pe.svm_log and (kernel == 'rbf'):
                    # trying out if some log-likelihood like representation is better for classification (makes sense when thinking that animals would have some profile modulating all probabilities)
                    eps = 1.e-16
                    for feature_ in features:
                        m_hist_1 = X_train[feature_][y_train==1, :].mean(axis=0) # average histogram for distractors x on the training set
                        for i_image in range(X_train[feature_].shape[0]): X_train[feature_][i_image, :] = np.log(X_train[feature_][i_image, :] + eps)-np.log(m_hist_1 + eps)
                        for i_image in range(X_test[feature_].shape[0]): X_test[feature_][i_image, :] = np.log(X_test[feature_][i_image, :] + eps)-np.log(m_hist_1 + eps)
                if self.pe.svm_norm:
                    if (kernel == 'rbf'):
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        for feature_ in features:
                            X_train[feature_] = scaler.fit_transform(X_train[feature_])
                            scaler.fit(X[feature_])
                            X_test[feature_] = scaler.transform(X_test[feature_])
                    else: # applying a "prior" (the probability represents the probability /knowing/ it belongs to a natural set)
                        eps = 1.e-16
                        for feature_ in features:
                            m_hist_1 = X_train[feature_][y_train==1, :].mean(axis=0) # average histogram for distractors x on the training set
                            for i_image in range(n_train):
                                X_train[feature_][i_image, :] = X_train[feature_][i_image, :]/(m_hist_1 + eps)
                                X_train[feature_][i_image, :] /= X_train[feature_][i_image, :].sum()
                            for i_image in range(n_test):
                                X_test[feature_][i_image, :] = X_test[feature_][i_image, :]/(m_hist_1 + eps)
                                X_test[feature_][i_image, :] /= X_test[feature_][i_image, :].sum()

                try:
                    # sanity check with a dummy classifier:
                    from sklearn.dummy import DummyClassifier
                    from sklearn import metrics
                    dc = DummyClassifier(strategy='most_frequent', random_state=0)
                    X_train_, X_test_ = np.zeros((n_train, 0)), np.zeros((n_test, 0))
                    for feature_ in features:
                        X_test_ = np.hstack((X_test_, X_test[feature_]))
                        X_train_ = np.hstack((X_train_, X_train[feature_]))
                    dc = dc.fit(X_train_, y_train)
                    self.log.warn("Sanity check with a dummy classifier:")
                    self.log.warn("score = %f ", dc.score(X_test_, y_test))#, scoring=metrics.f1_score))
                except Exception as e:
                    self.log.error("Failed doing the dummy classifier : %s ", e)
                ###############################################################################
                ###############################################################################
                # 3- preparing th gram matrix
                if not(kernel == 'rbf'):
                    # use KL distance as my kernel
                    kernel = 'precomputed'
                    def distance(x, y, KL_type=KL_type):
                        if KL_type=='sKL': return (self.KL(x, y) + self.KL(y, x))#symmetric KL
                        elif KL_type=='JSD': return (self.KL(x, (x+y)/2.) + self.KL(y, (x+y)/2.))/2.#Jensen-Shannon divergence
                        else: return self.KL(x, y)

                    def my_kernel(x, y, KL_m, use_log, KL_0):
                        d = distance(x, y, KL_type=KL_type)/KL_m
                        if use_log:
                            return d
                        else:
                            return np.exp(-d/KL_0)

                    n_train = X_train[feature_].shape[0]
                    n_test = X_test[feature_].shape[0]
                    gram_train = np.zeros((n_train, n_train))
                    gram_test = np.zeros((n_train, n_test))
                    for feature_ in features:
                        # compute the average KL
                        KL_0 = 0
                        for i_ in range(X_train[feature_].shape[0]):
                            for j_ in range(X_train[feature_].shape[0]):
                                KL_0 += distance(X_train[feature_][i_, :], X_train[feature_][j_, :], KL_type=KL_type)
                        KL_0 /= n_train**2
                        self.log.info('KL_0 = %f ', KL_0)
                    for feature_ in features:
                        for i_ in range(n_train):
                            for j_ in range(n_train):
                                gram_train[i_, j_] += my_kernel(X_train[feature_][i_, :], X_train[feature_][j_, :], KL_m=self.pe.svm_KL_m, use_log=self.pe.svm_log, KL_0=KL_0)

                        for i_ in range(n_train):
                            for j_ in range(n_test):
                                gram_test[i_, j_] += my_kernel(X_train[feature_][i_, :], X_test[feature_][j_, :], KL_m=self.pe.svm_KL_m, use_log=self.pe.svm_log, KL_0=KL_0)

                ###############################################################################
                # 4- Train a SVM classification model
                from sklearn.grid_search import GridSearchCV
                # see http://scikit-learn.org/stable/modules/grid_search.html
                from sklearn.svm import SVC
                from sklearn import cross_validation
                self.log.info("Fitting the classifier to the training set %s - %s - %s ",  databases, exp, feature)
                t0 = time.time()
                if kernel == 'precomputed':
                    C_range = np.logspace(self.pe.C_range_begin,self.pe.C_range_end, self.pe.N_svm_grid**2, base=2.)
                    gamma_range = np.logspace(self.pe.gamma_range_begin,self.pe.gamma_range_end, 1, base=2.)
                    param_grid = {'C': C_range }
                else:
                    C_range = np.logspace(self.pe.C_range_begin,self.pe.C_range_end, self.pe.N_svm_grid, base=2.)
                    gamma_range = np.logspace(self.pe.gamma_range_begin,self.pe.gamma_range_end, self.pe.N_svm_grid, base=2.)
                    param_grid = {'C': C_range, 'gamma': gamma_range }
                grid = GridSearchCV(SVC(verbose=False,
                                        kernel=kernel,
                                        tol=self.pe.svm_tol,
    #                                             probability=True,
                                        max_iter = self.pe.svm_max_iter,
                                        ),
                                    param_grid,
                                    verbose=1,
                                    scoring='f1',
                                    cv=5,
                                    n_jobs=self.pe.svm_n_jobs, # http://scikit-learn.org/0.13/modules/generated/sklearn.grid_search.GridSearchCV.html
                                    #pre_dispatch=2*self.pe.svm_n_jobs,
                                    )
                if kernel == 'precomputed':
                    grid.fit(gram_train, y_train)
                else:
                    X_train_ = np.zeros((n_train, 0))
                    for feature_ in features:
                        X_train_ = np.hstack((X_train_, X_train[feature_]))
                    grid.fit(X_train_, y_train)

                self.log.info("Fitting the classifier done in %0.3fs", (time.time() - t0))
                if self.log.level <= 10:
                    t0 = time.time()
                    self.log.info("Predicting the category names on the learning set")
                    if kernel == 'precomputed':
                        y_pred = grid.predict(gram_train)
                    else:
                        y_pred = grid.predict(X_train_)
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
                        ext = '.pdf'
                        figname = txtname.replace('.txt', '_grid' + ext)
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
                        fig.savefig(figname)
                        plt.close(fig)
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
                predicted_target.append(y_pred)
                if self.log.level<=10:
                    from sklearn.metrics import classification_report
                    print(classification_report(y_test, y_pred))
                    from sklearn.metrics import confusion_matrix
                    print(confusion_matrix(y_test, y_pred))
                # see https://en.wikipedia.org/wiki/F1_score
                try:
                    fone_score[i_cv] = np.array(metrics.f1_score(y_test, y_pred, labels=[0, 1], average=None)).mean()#'weighted')
                except:
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

                if edgeslist is None:# try: #(kernel=='rbf'):#len(databases)<3:
                    self.log.info(">> compiling results ")
                    t0 = time.time()
                    # tested_indices is the index of the image that is tested
                    # is_target is 1 if it is a target
                    # predicted_target is the response of the categorizer
                    imagelists = [self.get_imagelist(exp, name_database=databases[0]),
                                  self.get_imagelist(exp, name_database=databases[1])]
                    N_image = len(imagelists[0])
                    matname_score_dic = txtname.replace(self.pe.figpath, self.pe.matpath).replace('.txt', '.pickle')
                    try:
                        with open(matname_score_dic, "wb" ) as f:
                            results = pickle.load(f)
                    except:
                        results = {}
                        # setting up dictionary counting for each file how many times (integer) it is tested in total, how many times it is a target
                        for i_database in range(2):
                            for filename_, croparea_ in imagelists[i_database]:
                                results[filename_] = [0, 0]

#                     for vec in [tested_indices, is_target, predicted_target]: print len(vec)
                    for i_image, in_cat, pred_cat in zip(np.array(tested_indices).ravel(), np.array(is_target).ravel(), np.array(predicted_target).ravel()):
#                         print i_image, in_cat, pred_cat
                        filename_, croparea_ = imagelists[in_cat][i_image - in_cat*N_image]
                        results[filename_][0] +=  1 # how many times in total
                        results[filename_][1] +=  1*(pred_cat==1) # how many times it is a target
                    with open(matname_score_dic, "wb" ) as f:
                        pickle.dump(results, f)
                    self.log.info("Computing matname_score_dic done in %0.3fs", (time.time() - t0))
                t_cv = time.time()
                self.log.warn('Cross-validation in %s (%d/%d) - elaspsed = %0.1f s - ETA = %0.1f s ' % (matname_score, i_cv+1, self.pe.N_svm_cv, t_cv-t0_cv, (self.pe.N_svm_cv-i_cv-1)*(t_cv-t0_cv)/(i_cv+1) ) )

        try:
            np.save(matname_score, fone_score)
        except IOError as e:
            self.log.error('error %s while making %s ', e, matname_score)#, fone_score

        try:
            os.remove(matname_score + '_lock')
        except:
            self.log.error(' no matname_score lock file named %s_lock ', matname_score)

        return fone_score

    def compare(self, exp, databases=['serre07_distractors', 'serre07_targets'], noise=0., geometric=False, rho_quant=128, do_scale=True):
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
            imagelist, edgeslist, RMSE = self.process(exp, name_database=name_database, noise=noise)
#            edgeslist_db.append(edgeslist)
            if not(imagelist == 'locked'):
                self.log.info(' >> computing histogram for %s ', name_database)
                try:
                    v_hist_ = np.load(matname + '_kmeans_hist.npy')
                except Exception as e:
                    self.log.info(' >> There is no histogram, computing: %s ', e)
                    # images are separated in a learning and classification set: the histogram is computed on the first half of the images
                    N_image = edgeslist.shape[2]
                    v_hist_ = self.cohistedges(edgeslist[:, :, :N_image/2], display=None)
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
                    d_A = self.KL(v_hist_obs, v_hist[0])
                    d_B = self.KL(v_hist_obs, v_hist[1])
                    if geometric: rho_[ii_image] = d_A/np.sqrt(d_A**2+d_B**2)
                    else: rho_[ii_image] = d_A/np.sqrt(d_A*d_B)#/(d_A+d_B)
                rho.append(rho_)
            ha, dump = np.histogram(rho[0], np.linspace(0, 1., rho_quant), normed=True)#, density=True)
            hb, dump = np.histogram(rho[1], np.linspace(0, 1., rho_quant), normed=True)#, density=True)
            ha /= np.sum(ha)
            hb /= np.sum(hb)
            cdfa, cdfb = np.cumsum(ha), np.cumsum(hb)
            # plots
            fig = pylab.figure(figsize=(self.pe.figsize_hist, self.pe.figsize_hist))
            a = fig.add_subplot(111)#, polar = True)
            a.plot(cdfb, cdfa, color='r', lw=2)
            a.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), 'k--', lw=2)
            print(" >> AUC for experiment ", exp, " classifiying between databases ", databases, " = ", AUC(cdfb, cdfa))
            pylab.xlabel('false positive rate = 1 - Specificity')
            pylab.ylabel('false negative rate = Sensitivity')
            pylab.axis('tight')
            pylab.text(0.5, 0.1, 'AUC = ' + str(AUC(cdfb, cdfa)))
            fig.savefig(figname)
            pylab.close(fig)


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
