# -*- coding: utf8 -*-
from __future__ import division, print_function
"""
SparseEdges

See http://pythonhosted.org/SparseEdges

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
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

from LogGabor import LogGabor

class SparseEdges(LogGabor):
    def __init__(self, pe):
        """
        Initializes the SparseEdges class

        """
        LogGabor.__init__(self, pe)
        self.init()

    def init(self):
        LogGabor.init(self)

        self.n_levels = int(np.log(np.max((self.N_X, self.N_Y)))/np.log(self.pe.base_levels))
        self.sf_0 = 1. / np.logspace(1, self.n_levels, self.n_levels, base=self.pe.base_levels)
        self.theta = np.linspace(-np.pi/2, np.pi/2, self.pe.n_theta+1)[1:]

        self.oc = (self.N_X * self.N_Y * self.pe.n_theta * self.n_levels) #(1 - self.pe.base_levels**-2)**-1)
        for path in self.pe.figpath, self.pe.matpath, self.pe.edgefigpath, self.pe.edgematpath:
            if not(os.path.isdir(path)): os.mkdir(path)

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
        logD = np.zeros((self.N_X, self.N_Y, self.pe.n_theta, self.n_levels), dtype=np.complex)
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

    def init_C(self, image):
        C = np.empty((self.N_X, self.N_Y, self.pe.n_theta, self.n_levels), dtype=np.complex)
        for i_sf_0, sf_0 in enumerate(self.sf_0):
            for i_theta, theta in enumerate(self.theta):
                FT_lg = self.loggabor(0, 0, sf_0=sf_0, B_sf=self.pe.B_sf,
                                    theta=theta, B_theta=self.pe.B_theta)
                C[:, :, i_theta, i_sf_0] = self.FTfilter(image, FT_lg, full=True)
        return C

    def dipole(self, edge):

        y, x, theta_edge, sf_0, C, phase = edge # HACK
        theta_edge = np.pi/2 - theta_edge

        D = np.ones((self.N_X, self.N_Y, self.pe.n_theta, self.n_levels))
        distance = np.sqrt(((1.*self.X-x)**2+(1.*self.Y-y)**2)/(self.N_X**2+self.N_Y**2))/self.pe.dip_w
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
        image = np.zeros((self.N_X, self.N_Y))
#        print edges.shape, edges[:, 0]
        for i_edge in range(edges.shape[1]):#self.pe.N):
            # TODO : check that it is correct when we remove alpha when making new MP
            if not mask or ((edges[0, i_edge]/self.N_X -.5)**2+(edges[1, i_edge]/self.N_Y -.5)**2) < .5**2:
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
            fig = plt.figure(figsize=(self.pe.figsize_edges*self.N_Y/self.N_X, self.pe.figsize_edges))
        if a==None:
            border = 0.0
            a = fig.add_axes((border, border, 1.-2*border, 1.-2*border), axisbg='w')
        a.axis(c='b', lw=0, frame_on=False)

        if color == 'black' or color == 'redblue' or color in['brown', 'green', 'blue']: #cocir or chevrons
            linewidth = self.pe.line_width_chevrons
            scale = self.pe.scale_chevrons
        else:
            linewidth = self.pe.line_width
            scale = self.pe.scale

        opts= {'extent': (0, self.N_Y, self.N_X, 0), # None, #
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
#             X, Y, Theta, Sf_0 = edges[1, :]+.5, self.N_X - edges[0, :]-.5, edges[2, :], edges[3, :] # HACK in orientation
            weights = edges[4, :]
            weights = weights/(np.abs(weights)).max()
            phases = edges[5, :]

            for x, y, theta, sf_0, weight, phase in zip(X, Y, Theta, Sf_0, weights, phases):
                if not mask or ((x/self.N_X -.5)**2+(y/self.N_Y -.5)**2) < .5**2:
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
            circ = plt.Circle((.5*self.N_Y, .5*self.N_Y), radius=0.5*self.N_Y-linewidth/2., fill=False, facecolor='none', edgecolor = 'black', alpha = 0.5, ls='dashed', lw=linewidth)
            a.add_patch(circ)
        a.axis([0, self.N_Y, self.N_X, 0])
        a.grid(b=False, which="both")
        plt.draw()
        if mappable:
            return fig, a, line_segments
        else:
            return fig, a

    def texture(self, N_edge = 256, filename='', croparea='', randn=True):
        # a way to get always the same seed for each image
        if not (filename==''):# or not (croparea==''):
            np.random.seed(seed=int(int("0x" +  hashlib.sha224((filename+str(croparea)).encode('utf-8')).hexdigest(), 0)*1. % 4294967295))
        # white noise or texture
        if randn:
            return np.random.randn(self.N_X, self.N_Y)
        else:
            edgeslist = np.zeros((6, N_edge))
            edgeslist[0, :] = self.N_X * np.random.rand(N_edge)
            edgeslist[1, :] = self.N_X * np.random.rand(N_edge)
            edgeslist[2, :] = (np.pi* np.random.rand(N_edge) ) % np.pi
            edgeslist[3, :] =  self.sf_0[np.random.randint(self.sf_0.size, size=(N_edge))] # best would be to have more high frequency components
            edgeslist[4, :] = np.random.randn(N_edge)
            edgeslist[5, :] = 2*np.pi*np.random.rand(N_edge)
            image_rec = self.reconstruct(edgeslist)
            image_rec /= image_rec.std()
            return image_rec

    def full_run(self, exp, name_database, imagelist, noise):
        """
        runs the edge extraction for a list of images

        """
        N_do = 2
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
                    time.sleep(1.*np.random.rand())
                    if not(os.path.isfile(matname + '_lock')):
                        self.log.info('Doing edge extraction of %s ', matname)
                        open(matname + '_lock', 'w').close()
                        image, filename_, croparea_ = self.patch(name_database, filename=filename, croparea=croparea)
                        if noise > 0.: image += noise*image[:].std()*self.texture(filename=filename, croparea=croparea)
                        edges, C = self.run_mp(image)
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
                self.log.error(' some locked edge extractions %s, error ', e)
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
                        image_ = image.copy()
                        image_rec = np.zeros_like(image_)
                        if self.pe.do_whitening: image_ = self.whitening(image_)
                        for i_N in range(self.pe.N):
                            image_rec += self.reconstruct(edges[:, i_N][:, np.newaxis])
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
        self.edges_sf_0 = 2**np.arange(np.ceil(np.log2(self.N_X)))
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
            mask = ((x/self.N_X -.5)**2+(y/self.N_Y -.5)**2) < .5**2
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
            mask = ((x/self.N_X -.5)**2+(y/self.N_Y -.5)**2) < .5**2
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

        if not(edgeslist==None):
            v_hist = None
            six, N_edge, N_image = edgeslist.shape
            # TODO: vectorize over images?
            for i_image in range(N_image):
                # retrieve individual positions, orientations, scales and coefficients
                X, Y = edgeslist[0, :, i_image], edgeslist[1, :, i_image]
                Theta = edgeslist[2, :, i_image]
                Sf_0 = edgeslist[3, :, i_image]
                value = edgeslist[4, :, i_image]
                phase = edgeslist[5, :, i_image]
                if self.pe.edge_mask:
                    # remove edges whose center position is not on the central disk
                    mask = ((X/self.N_X -.5)**2+(Y/self.N_Y -.5)**2) < .5**2
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
                d = np.sqrt(dx**2 + dy**2) / self.N_X  # distance normalized by the image size
                # TODO: check that we correctly normalize position by the scale of the current edge
                if self.pe.scale_invariant: d *= np.sqrt(Sf_0[:, np.newaxis]*Sf_0[np.newaxis, :])#*np.sqrt(self.N_X)
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
                else:
                    weights = 1.
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
                                                 weights=weights.ravel()
                                                )
#                 print v_hist_.sum(), v_hist_.min(), v_hist_.max(), d.ravel().shape
                if v_hist_.sum()<.01: self.log.debug(' less than 1 percent of co-occurences within ranges: %f ', v_hist_.sum())
                if not(v_hist_.sum() == 0.):
                    # add to the full histogram
                    if v_hist is None:
                        v_hist = v_hist_*1.
                    else:
                        v_hist += v_hist_*1.
        if v_hist is None or (v_hist.sum() == 0.):
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
                        rad = d_r / self.pe.d_max * max(self.N_X, self.N_Y) /2
                        ii_phi = i_r * self.pe.N_phi
                        colin_edgelist[0:2, ii_phi + i_phi] =  self.N_X /2 - rad * np.sin(phi + np.pi/self.pe.N_phi/2), self.N_Y /2 + rad * np.cos(phi + np.pi/self.pe.N_phi/2)
                        colin_edgelist[2, ii_phi + i_phi] = self.edges_theta[colin_argmax[i_r, i_phi]] + np.pi/self.pe.N_Dtheta/2
                        colin_edgelist[3, ii_phi + i_phi] = edge_scale
                        colin_edgelist[4, ii_phi + i_phi] = v_hist_noscale[i_r, i_phi, colin_argmax[i_r, i_phi]]
                        # symmetric
                        colin_edgelist[:, ii_phi + i_phi +  self.pe.N_r * self.pe.N_phi] = colin_edgelist[:, ii_phi + i_phi]
                        colin_edgelist[0:2, ii_phi + i_phi +  self.pe.N_r * self.pe.N_phi] = self.N_X - colin_edgelist[0, ii_phi + i_phi], self.N_Y - colin_edgelist[1, ii_phi + i_phi]
                # reference angle
                colin_edgelist[:, -1] = [self.N_X /2, self.N_Y /2, 0, edge_scale, colin_edgelist[4,:].max() *1.2, 0.]
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
                        rad = d_r / self.pe.d_max * max(self.N_X, self.N_Y) /2
                        ii_theta = i_r * self.pe.N_Dtheta
                        cocir_edgelist[0:2, ii_theta + i_theta] =  self.N_X /2 - rad * np.sin( self.edges_phi[cocir_proba[i_r, i_theta]] + np.pi/self.pe.N_phi/2), self.N_Y /2 + rad * np.cos( self.edges_phi[cocir_proba[i_r, i_theta]] + np.pi/self.pe.N_phi/2)
                        cocir_edgelist[2, ii_theta + i_theta] = theta + np.pi/self.pe.N_Dtheta/2
                        cocir_edgelist[3, ii_theta + i_theta] = edge_scale
                        cocir_edgelist[4, ii_theta + i_theta] = v_hist_noscale[i_r, cocir_proba[i_r, i_theta], i_theta]
                        # symmetric
                        cocir_edgelist[:, ii_theta + i_theta +  self.pe.N_r * self.pe.N_Dtheta] = cocir_edgelist[:,  ii_theta + i_theta]
                        cocir_edgelist[0:2, ii_theta + i_theta +  self.pe.N_r * self.pe.N_Dtheta] = self.N_X - cocir_edgelist[0,  ii_theta + i_theta], self.N_Y - cocir_edgelist[1, ii_theta + i_theta]
                cocir_edgelist[:, -1] = [self.N_X /2, self.N_Y /2, 0, edge_scale, cocir_edgelist[4,:].max() *1.2, 0.]
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
            rad_X, rad_Y = 1.* self.N_X/s_theta, 1.*self.N_Y/s_phi
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
                                       self.N_X - rad_X * (s_theta - i_theta - .5) + .5),
#                                             facecolor=fc, edgecolor=fc,
                                       rad, lw=self.pe.line_width_chevrons/2)
                    mypatches.append(circ)
                    colors.append(value*score)

                    # first edge
#                    print i_phi, i_theta,  s_phi, s_theta, v_hist_angle.shape
                    angle_edgelist[0, i_phi * s_theta + i_theta] = self.N_X - rad_X * (s_theta - i_theta - .5)
                    angle_edgelist[1, i_phi * s_theta + i_theta] = rad_Y * (i_phi + .5) - rad * 1.
                    angle_edgelist[2, i_phi * s_theta + i_theta] = phi + theta/2
                    angle_edgelist[3, i_phi * s_theta + i_theta] = self.pe.edge_scale_chevrons
                    angle_edgelist[4, i_phi * s_theta + i_theta] = 1.
                    # second edge
                    angle_edgelist[0, i_phi * s_theta + i_theta + s_phi * s_theta] = self.N_X - rad_X * (s_theta - i_theta - .5)
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
            p = PatchCollection(mypatches, norm=MidpointNormalize(midpoint=0, vmin=v_min, vmax=v_max), cmap=cm.coolwarm, alpha=1.0)
            p.set_array(np.array(colors))
            if dolog:
                p.set_clim([v_min, v_max])
            else:
                p.set_clim([v_min, v_max])
            a.add_collection(p)

#            print rad/s_theta, rad/s_phi
            fig, a = self.show_edges(angle_edgelist, fig=fig, a=a, image=None, color='black')

            if colorbar:
                cbar = plt.colorbar(ax=a, mappable=p, shrink=0.6)
                if dolog:
                    if cbar_label: cbar.set_label('probability ratio')
                    ticks_cbar = 2**(np.floor(np.linspace(v_min, v_max, 5)))
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
                eps = 0.55 # HACK to center grid. dunnon what's happening here
                if half:
                    plt.setp(a, xticks=[(1./self.pe.N_phi/1.25)*self.N_X, (1. - 1./self.pe.N_phi/1.25)*self.N_X])
                    if not(xticks=='left'):
                        plt.setp(a, xticklabels=[r'$0$', r'$\frac{\pi}{2}$'])
                    else:
                        plt.setp(a, xticklabels=[r'', r''])
                else:
                    plt.setp(a, xticks=[(1./(self.pe.N_phi+1)/2)*self.N_X, .5*self.N_X+eps, (1. - 1./(self.pe.N_phi+1)/2)*self.N_X])
                    if not(xticks=='left'):
                        plt.setp(a, xticklabels=[r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$'])
                    else:
                        plt.setp(a, xticklabels=[r'', r''])
                if half:
                    plt.setp(a, yticks=[(1./self.pe.N_Dtheta)*self.N_Y, (1. - 1./(self.pe.N_Dtheta+.45))*self.N_Y])
                    if not(xticks=='bottom'):
                        plt.setp(a, yticklabels=[r'$0$', r'$\frac{\pi}{2}$'])
                    else:
                        plt.setp(a, yticklabels=[r'', r''])
                else:
                    plt.setp(a, yticks=[1./(self.pe.N_Dtheta+1)/2*self.N_X, .5*self.N_Y+eps, (1. - 1./(self.pe.N_Dtheta+1)/2)*self.N_Y])
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
                    if RMSE == 'locked':
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
        if True:
            v_hist /= v_hist.sum()
            v_hist_obs /= v_hist_obs.sum()
            # taking advantage of log(True) = 0 and canceling out null bins in v_hist_obs
            return np.sum(v_hist.ravel()*(np.log(v_hist.ravel()+(v_hist == 0).ravel())
                                        - np.log(v_hist_obs.ravel()+(v_hist_obs == 0).ravel())))
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

    def plot(self, mps, experiments, databases, labels, fig=None, ax=None, color=[1., 0., 0.], threshold=None, scale=False, ref=None):
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
                if True: #try:
                    imagelist, edgeslist, RMSE = mp.process(exp=experiment, name_database=name_database)
                    RMSE /= RMSE[:, 0][:, np.newaxis]
                    N = RMSE.shape[1] #number of edges
                    l0_max = max(l0_max, N*np.log2(mp.oc)/mp.N_X/mp.N_Y)
                    if not(scale):
                        l0_axis = np.arange(N)
                    else:
                        l0_axis = np.linspace(0, N*np.log2(mp.oc)/mp.N_X/mp.N_Y, N)
                    errorevery_zoom = 1.4**(1.*eev/len(experiments))
                    errorevery = np.max((int(RMSE.shape[1]/8*errorevery_zoom), 1))
                    ax.errorbar(l0_axis, RMSE.mean(axis=0),
                                yerr=RMSE.std(axis=0), errorevery=errorevery)
                    inset.errorbar(l0_axis, edgeslist[4, :, :].mean(axis=1),
                                yerr=edgeslist[4, :, :].std(axis=1), label=label, errorevery=errorevery)
                    ax.plot(l0_axis[::errorevery], RMSE.mean(axis=0)[::errorevery],
                                linestyle='None', marker='o', ms=3)
                    inset.plot(l0_axis[::errorevery], edgeslist[4, :, :].mean(axis=1)[::errorevery],
                                linestyle='None',  marker='o', ms=3)
                    eev += 1
                #except Exception as e:
                #    print('Failed to plot experiment %s with error : %s ' % (experiment, e) )
            ax.set_ylim([.0, 1.02])
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
                a.spines['left'].set_smart_bounds(True)
                a.spines['bottom'].set_smart_bounds(True)
                a.xaxis.set_ticks_position('bottom')
                a.yaxis.set_ticks_position('left')
                if not(scale):#False and a==ax:
                    a.set_xlabel(r'$\ell_0$-norm')
                else:
                    ax.set_xlabel(r'relative $\ell_0$ pseudo-norm (bits / pixel)')#relative $\ell_0$-norm')

                a.grid(b=False, which="both")

            ax.set_ylabel(r'Squared error')
            inset.set_ylabel(r'Coefficient')
            inset.legend(loc='upper right', frameon=False)#, bbox_to_anchor = (0.5, 0.5))
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
                        l0_results *= np.log2(mp.oc)/mp.N_X/mp.N_Y
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
            relSE, relSE_std = [], []
            imagelist_ref, edgeslist_ref, RMSE_ref = mps[ref].process(exp=experiments[ref], name_database=databases[ref])
            RMSE_ref /= RMSE_ref[:, 0][:, np.newaxis]
            l0_ref = np.log2(mps[ref].oc)/mps[ref].N_X/mps[ref].N_Y

            for mp, experiment, name_database, label in zip(mps, experiments, databases, labels):
                try:
                    imagelist, edgeslist, RMSE = mp.process(exp=experiment, name_database=name_database)
                    RMSE /= RMSE[:, 0][:, np.newaxis]
                    N = RMSE.shape[1] #number of edges
                    l0 = np.log2(mp.oc)/mp.N_X/mp.N_Y
                    relSE.append((RMSE/RMSE_ref).mean())
                    relSE_std.append((RMSE/RMSE_ref).std(axis=0).mean())
                except Exception as e:
                    print('Failed to plot experiment %s with error : %s ' % (experiment, e) )
            fig.subplots_adjust(wspace=0.1, hspace=0.1,
                                left=0.2, right=0.9,
                                top=0.9,    bottom=0.175)

            ind = len(relSE)
            width = .8
            ax.bar(np.arange(ind), relSE, yerr=relSE_std)
            ax.set_xlim([-width/4, ind+.0*width])

            if not(scale):#False and a==ax:
                ax.set_ylabel(r'SE')
            else:
#             ax.set_ylabel(r'relative $\ell_0$ pseudo-norm')# (bits / pixel)')#relative $\ell_0$-norm')
                ax.set_ylabel(r'rel. SE')# (bits / pixel)')#relative $\ell_0$-norm')

            ax.set_xticks(np.arange(ind)+.5*width)
            ax.set_xticklabels(labels)

            plt.tight_layout()
#         fig.set_tight_layout(True)

            return fig, ax, ax

        else: # fourth type: we have a reference and a threshold
            relL0, relL0_std = [], []
            # computes for the reference 
            imagelist_ref, edgeslist_ref, RMSE_ref = mps[ref].process(exp=experiments[ref], name_database=databases[ref])
            RMSE_ref /= RMSE_ref[:, 0][:, np.newaxis] # normalize RMSE
            L0_ref =  np.argmax(RMSE_ref<threshold, axis=1)*1. +1
            if scale: L0_ref *= np.log2(mps[ref].oc)/mps[ref].N_X/mps[ref].N_Y
#             print("ref-thr - L0_ref=", L0_ref)

            for mp, experiment, name_database, label in zip(mps, experiments, databases, labels):
                try:
                    imagelist, edgeslist, RMSE = mp.process(exp=experiment, name_database=name_database)
                    RMSE /= RMSE[:, 0][:, np.newaxis] # normalize RMSE
                    N = RMSE.shape[1] #number of edges
                    L0 =  np.argmax(RMSE<threshold, axis=1)*1.
                    if RMSE.min()>threshold: print('the threshold is never reached for', experiment, name_database)
                    if scale: L0 *= np.log2(mp.oc)/mp.N_X/mp.N_Y
                    relL0.append((L0/L0_ref).mean())
                    relL0_std.append((L0/L0_ref).std())
                except Exception as e:
                    print('Failed to analyze experiment %s with error : %s ' % (experiment, e) )
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
            im_RGB = np.zeros((self.N_X, self.N_Y, 3))
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
