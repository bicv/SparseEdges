# -*- coding: utf8 -*-
"""
SparseEdges

See http://pythonhosted.org/SparseEdges

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
import numpy as np
import scipy.ndimage as nd
import os
PID, HOST = os.getpid(), os.uname()[1]
TAG = 'host-' + HOST + '_pid-' + str(PID)
# -------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
import sys, traceback
import logging
logging.basicConfig(filename='log-sparseedges-debug.log', format='%(asctime)s@[' + TAG + '] %(message)s', datefmt='%Y%m%d-%H:%M:%S')
log = logging.getLogger("SparseEdges")
#log.setLevel(level=logging.WARN)
log.setLevel(level=logging.INFO)
# log.setLevel(logging.DEBUG) #set verbosity to show all messages of severity >= DEBUG
import matplotlib.pyplot as plt

class SparseEdges:
    def __init__(self, lg):
        """
        initializes the LogGabor structure

        """
        self.pe = lg.pe
        self.MP_alpha = self.pe.MP_alpha

        self.lg = lg
        self.im = lg.im
        self.N_X = lg.N_X
        self.N_Y = lg.N_Y

        self.base_levels = self.pe.base_levels
        self.n_levels = int(np.log(np.max((self.N_X, self.N_Y)))/np.log(self.base_levels))
        self.sf_0 = 1. / np.logspace(1, self.n_levels, self.n_levels, base=self.base_levels)

        self.n_theta = self.pe.n_theta
        self.theta_ = np.linspace(0., np.pi, self.n_theta, endpoint=False)
        self.B_theta = self.pe.B_theta
        self.B_sf = self.pe.B_sf

        self.N = self.pe.N
        self.do_whitening = self.pe.do_whitening
        self.do_mask = self.pe.do_mask
        self.MP_alpha = self.pe.MP_alpha
        if self.do_mask:
            X, Y = np.mgrid[-1:1:1j*self.N_X, -1:1:1j*self.N_Y]
            self.mask = (X**2 + Y**2) < 1.
        for path in self.pe.figpath, self.pe.matpath, self.pe.edgefigpath, self.pe.edgematpath:
            if not(os.path.isdir(path)): os.mkdir(path)

    def run_mp(self, image, verbose=False):
        """
        runs the MatchingPursuit algorithm on image

        """
        edges = np.zeros((6, self.N))
        image_ = image.copy()
        if self.do_whitening: image_ = self.im.whitening(image_)
        C = self.init(image_)
        for i_edge in range(self.N):
            # MATCHING
            ind_edge_star = self.argmax(C)
            # recording
            if verbose: print 'Max activity  : ', np.absolute(C[ind_edge_star]), ' phase= ', np.angle(C[ind_edge_star], deg=True), ' deg,  @ ', ind_edge_star
            edges[:, i_edge] = np.array([ind_edge_star[0]*1., ind_edge_star[1]*1.,
                                         self.theta_[ind_edge_star[2]],
                                         self.sf_0[ind_edge_star[3]],
                                         self.MP_alpha * np.absolute(C[ind_edge_star]), np.angle(C[ind_edge_star])])
            # PURSUIT
            C = self.backprop(C, ind_edge_star)
#            if verbose: print 'Residual activity : ',  C[ind_edge_star]
        return edges, C
#
    def init(self, image):
        C = np.empty((self.N_X, self.N_Y, self.n_theta, self.n_levels), dtype=np.complex)
        for i_sf_0, sf_0 in enumerate(self.sf_0):
            for i_theta, theta in enumerate(self.theta_):
                FT_lg = self.lg.loggabor(0, 0, sf_0=sf_0, B_sf=self.B_sf,
                                    theta=theta, B_theta=self.B_theta)
                C[:, :, i_theta, i_sf_0] = self.im.FTfilter(image, FT_lg, full=True)
                if self.do_mask: C[:, :, i_theta, i_sf_0] *= self.mask
        return C

    def argmax(self, C):
        """
        Returns the ArgMax from C by returning the
        (x_pos, y_pos, theta, scale)  tuple

        >>> C = np.random.randn(10, 10, 5, 4)
        >>> C[x_pos][y_pos][level][level] = C.max()

        """
        ind = np.absolute(C).argmax()
        return np.unravel_index(ind, C.shape)

    def backprop(self, C, ind_edge_star):
        """
        Removes edge_star from the activity

        """
        C_star = self.MP_alpha * C[ind_edge_star]
        FT_lg_star = self.lg.loggabor(ind_edge_star[0]*1., ind_edge_star[1]*1.,
                                      theta=self.theta_[ind_edge_star[2]], B_theta=self.B_theta,
                                      sf_0=self.sf_0[ind_edge_star[3]], B_sf=self.B_sf,
                                      )
        lg_star = self.im.invert(C_star*FT_lg_star, full=False)

        for i_sf_0, sf_0 in enumerate(self.sf_0):
            for i_theta, theta in enumerate(self.theta_):
                FT_lg = self.lg.loggabor(0, 0, sf_0=sf_0, B_sf=self.B_sf, theta=theta, B_theta=self.B_theta)
                C[:, :, i_theta, i_sf_0] -= self.im.FTfilter(lg_star, FT_lg, full=True)
                if self.do_mask: C[:, :, i_theta, i_sf_0] *= self.mask
        return C

    def reconstruct(self, edges):
        image = np.zeros((self.N_X, self.N_Y))
#        print edges.shape, edges[:, 0]
        for i_edge in range(edges.shape[1]):#self.N):
            # TODO : check that it is correct when we remove alpha when making new MP
            image += self.im.invert(edges[4, i_edge] * np.exp(1j*edges[5, i_edge]) *
                                    self.lg.loggabor(
                                                    edges[0, i_edge], edges[1, i_edge],
                                                    theta=edges[2, i_edge], B_theta=self.B_theta,
                                                    sf_0=edges[3, i_edge], B_sf=self.B_sf,
                                                    ),
                                    full=False)
        return image

    def adapt(self, edges):
        # TODO : implement a COMP adaptation of the thetas and scales tesselation of Fourier space
        pass

    def show_edges(self, edges, fig=None, a=None, image=None, norm=True,
                   color='auto', v_min=-1., v_max=1., show_phase=False, gamma=1., pedestal=.2, mappable=False):
        """
        Shows the quiver plot of a set of edges, optionally associated to an image.

        """
        import matplotlib.cm as cm
        if fig==None:
            fig = plt.figure(figsize=(self.pe.figsize_edges, self.pe.figsize_edges))
        if a==None:
            border = 0.0
            a = fig.add_axes((border, border, 1.-2*border, 1.-2*border), axisbg='w')
        a.axis(c='b', lw=0)

        if color == 'black' or color == 'redblue' or color in['brown', 'green', 'blue']: #cocir or chevrons
            linewidth = self.pe.line_width_chevrons
            scale = self.pe.scale_chevrons
        else:
            linewidth = self.pe.line_width
            scale = self.pe.scale

        opts= {'extent': (0, self.N_X, self.N_Y, 0),
               'cmap': cm.gray,
               'vmin':v_min, 'vmax':v_max, 'interpolation':'nearest', 'origin':'upper'}
#         origin : [‘upper’ | ‘lower’], optional, default: None
#         Place the [0,0] index of the array in the upper left or lower left corner of the axes. If None, default to rc image.origin.
#         extent : scalars (left, right, bottom, top), optional, default: None
#         Data limits for the axes. The default assigns zero-based row, column indices to the x, y centers of the pixels.
        if not(image == None):
#             if image.ndim==2: opts['cmap'] = cm.gray
            if norm: image = self.im.normalize(image, center=True, use_max=True)
            a.imshow(image, **opts)
        else:
            a.imshow([[v_max]], **opts)
        if edges.shape[1] > 0:
            from matplotlib.collections import LineCollection#, EllipseCollection
            import matplotlib.patches as patches
            # draw the segments
            segments, colors, linewidths = list(), list(), list()

            X, Y, Theta, Sf_0 = edges[1, :]+.5, edges[0, :]+.5, np.pi -  edges[2, :], edges[3, :]
            weights = edges[4, :]

            #show_phase, pedestal = False, .2 # color edges according to phase or hue? pedestal value for alpha when weights= 0

    #        print X, Y, Theta, Sf_0, weights, scale_
    #        print 'Min theta ', Theta.min(), ' Max theta ', Theta.max()
#            weights = np.absolute(weights)/(np.abs(weights)).max()
            weights = weights/(np.abs(weights)).max()

            for x, y, theta, sf_0, weight in zip(X, Y, Theta, Sf_0, weights):
                u_, v_ = np.cos(theta)*scale/sf_0*self.N_X, np.sin(theta)*scale/sf_0*self.N_Y
                segment = [(x - u_, y - v_), (x + u_, y + v_)]
                segments.append(segment)
                if color=='auto':
                    if show_phase:
                        #colors.append(cm.hsv(np.angle(weight), alpha=pedestal + (1. - pedestal)*weight**gamma))#))
                        colors.append(cm.hsv(0., alpha=pedestal + (1. - pedestal)*weight**gamma))#)) # HACK
                    else: colors.append(cm.hsv((theta % np.pi)/np.pi, alpha=pedestal + (1. - pedestal)*weight))#))
                elif color == 'black':
                    colors.append((0, 0, 0, 1))# black
                elif color == 'green': # figure 1DE
                    colors.append((0.05, 0.5, 0.05, np.abs(weight)**gamma))
                elif color == 'blue': # figure 1DE
                    colors.append((0.05, 0.05, 0.5, np.abs(weight)**gamma))
                elif color == 'brown': # figure 1DE
                    colors.append((0.5, 0.05, 0.05, np.abs(weight)**gamma))
                else: # geisler maps etc...
                    colors.append(((np.sign(weight)+1)/2, 0, (1-np.sign(weight))/2, np.abs(weight)**gamma))#weight*(1-weight)))# between red and blue
                linewidths.append(linewidth) # *weight thinning byalpha...

            # TODO : put circle in front
            for x, y, theta, sf_0, weight in zip(X, Y, Theta, Sf_0, weights):
                if color=='auto':
                    if show_phase:
                        #fc = cm.hsv(np.angle(weight), alpha=pedestal + (1. - pedestal)*weight**gamma)
                        fc = cm.hsv(0., alpha=pedestal + (1. - pedestal)*weight**gamma) # HACK
                    else:
                        fc = cm.hsv((theta % np.pi)/np.pi, alpha=pedestal + (1. - pedestal)*weight**gamma)
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
                # http://matplotlib.sourceforge.net/users/transforms_tutorial.html
                circ = patches.Circle((x,y), self.pe.scale_circle*scale/sf_0, facecolor=fc, edgecolor='none')#, alpha=0.5*weight)
                # (0.5, 0.5), 0.25, transform=ax.transAxes, facecolor='yellow', alpha=0.5)
                a.add_patch(circ)

            line_segments = LineCollection(segments, linewidths=linewidths, colors=colors, linestyles='solid')
            a.add_collection(line_segments)

        if not(color=='auto'):# chevrons maps etc...
            plt.setp(a, xticks=[])
            plt.setp(a, yticks=[])

        a.axis([0, self.N_X, self.N_Y, 0])
        plt.draw()
        if mappable:
            return fig, a, line_segments
        else:
            return fig, a

    def full_run(self, exp, name_database, imagelist, noise):
        """
        runs the edge extraction for a list of images

        """
        N_image = len(imagelist)

        global_lock = False # switch to True when we resume a batch and detect that one edgelist is not finished in another process
        for filename, croparea in imagelist:
#                 signal = do_edge(self, image, exp, name_database, filename, croparea)
#                         def do_edge(self, image, exp, name_database, filename, croparea):
            matname = os.path.join(self.pe.edgematpath, exp + '_' + name_database, filename + str(croparea) + '.npy')
            if not(os.path.isdir(os.path.join(self.pe.edgematpath, exp + '_' + name_database))):
                os.mkdir(os.path.join(self.pe.edgematpath, exp + '_' + name_database))
            if not(os.path.isfile(matname)):
                if not(os.path.isfile(matname + '_lock')):
                    file(matname + '_lock', 'w').close()
                    image, filename_, croparea_ = self.im.patch(name_database, filename=filename, croparea=croparea)
                    if noise > 0.: image += noise*image[:].std()*np.random.randn(image.shape[0], image.shape[1])
                    edges, C = self.run_mp(image)
                    np.save(matname, edges)
                    try:
                        os.remove(matname + '_lock')
                    except Exception, e:
                        log.error('Failed to remove lock file %s_lock, error : %s ', matname, traceback.print_tb(sys.exc_info()[2]))
                else:
                    log.info('The edge extraction at step %s is locked', matname)
                    global_lock = True

        if global_lock is True:
            log.error(' some locked edge extractions ')
            return 'locked'
        else:
            try:
                edgeslist = np.zeros((6, self.N, N_image))
                i_image = 0
                for filename, croparea in imagelist:
                    matname = os.path.join(self.pe.edgematpath, exp + '_' + name_database, filename + str(croparea) + '.npy')
                    edgeslist[:, :, i_image] = np.load(matname)
                    i_image += 1
                return edgeslist
            except:
                log.error(' some locked edge extractions ')
                return 'locked'

    def init_edges(self):
        # configuring histograms
        # sequence of scalars,it defines the bin edges, including the rightmost edge.
        self.edges_d = np.linspace(self.pe.d_min, self.pe.d_max, self.pe.N_r+1)
        self.edges_phi = np.linspace(-np.pi/2, np.pi/2, self.pe.N_phi+1) + np.pi/self.pe.N_phi/2
        self.edges_theta = np.linspace(-np.pi/2, np.pi/2, self.pe.N_Dtheta+1) + np.pi/self.pe.n_theta/2
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

        weights = np.absolute(value)/(np.absolute(value)).sum()
        v_hist, v_theta_edges_ = np.histogram(theta, self.edges_theta, normed=True, weights=weights)
        v_hist /= v_hist.sum()
        if display:
            if fig==None: fig = plt.figure(figsize=(self.pe.figsize_hist, self.pe.figsize_hist))
            if a==None: a = plt.axes(polar=True, axisbg='w')
            a.bar(self.edges_theta[:-1], v_hist, width=np.pi/self.pe.N_Dtheta, color='b')# edgecolor="none")
            a.bar(self.edges_theta[:-1]+np.pi, v_hist, width=np.pi/self.pe.N_Dtheta, color='g')
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
        v_hist, v_sf_0_edges_ = np.histogram(sf_0, self.edges_sf_0, normed=True, weights=weights)
        v_hist /= v_hist.sum()
        if display:
            if fig==None: fig = plt.figure(figsize=(self.pe.figsize_hist, self.pe.figsize_hist))
            if a==None: a = fig.add_subplot(111, axisbg='w')
            a.bar(v_sf_0_edges_[:-1], v_hist)
            plt.setp(a, yticks=[])
            plt.xlabel('SF_0')
            plt.ylabel('probability')
            return fig,a
        else:
            return v_hist, v_theta_edges_

    def cohistedges(self, edgeslist, v_hist=None, prior=None,
                    fig=None, a=None, symmetry=True,
                    display='chevrons', v_min=None, v_max=None, labels=True, mappable=False, radius=None,
                    xticks=False, half=False, dolog=False, color='redblue', colorbar=True, cbar_label=True):
        """
        second-order stats= center all edges around the current one by rotating and scaling

        p(x-x_, y-y_, theta-theta_ | I, x_, y_, theta_)

        """
        self.init_edges()

        if not(edgeslist==None):
            v_hist = None
            five, N_edge, N_image = edgeslist.shape
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
                                                 normed=True,
                                                 weights=weights.ravel()
                                                )
#                 print v_hist_.sum(), v_hist_.min(), v_hist_.max(), d.ravel().shape
                if v_hist_.sum()<.8: log.error(' less than 80 percent of co-occurences within ranges: %f ', v_hist_.sum())
                if not(v_hist_.sum() == 0.):
                    # add to the full histogram
                    if v_hist is None:
                        v_hist = v_hist_*1.
                    else:
                        v_hist += v_hist_*1.
        if v_hist is None or (v_hist.sum() == 0.):
            v_hist = np.ones(v_hist_.shape)

        v_hist /= v_hist.sum()

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
                colin_edgelist[:, -1] = [self.N_X /2, self.N_Y /2, 0, edge_scale, colin_edgelist[4,:].max() *1.2 ]
                return self.show_edges(colin_edgelist, fig=fig, a=a, image=None, v_min=0., v_max=v_hist_noscale.max(), color=color)
            except Exception, e:
                log.error(' failed to generate colin_geisler plot, %s', traceback.print_tb(sys.exc_info()[2]))
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
                cocir_edgelist[:, -1] = [self.N_X /2, self.N_Y /2, 0, edge_scale, cocir_edgelist[4,:].max() *1.2 ]
                return self.show_edges(cocir_edgelist, fig=fig, a=a, image=None, v_min=0., v_max=v_hist_noscale.max(), color=color)
            except Exception, e:
                log.error(' failed to generate cocir_geisler plot, %s', traceback.print_tb(sys.exc_info()[2]))
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
                log.error(' failed to generate cohist_scale, %s', e)
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
            import matplotlib.pyplot as plt
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
                def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False, gamma=1.3):
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
            p = PatchCollection(mypatches, norm=MidpointNormalize(midpoint=0, vmin=v_min, vmax=v_max), cmap=cm.coolwarm, alpha=0.8)
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


    def process(self, exp, note='', name_database='natural', noise=0.):
        """
        The pipeline to go from one database to a list of edge lists

        ``note`` designs a string that modified the histogram (such as changing the number of bins)

        """

        log.info(' > computing edges for experiment %s with database %s ', exp, name_database)
        #: 1 - Creating an image list
        locked = False
        matname = os.path.join(self.pe.edgematpath, exp + '_' + name_database)
        #while os.path.isfile(matname + '_images_lock'):
        time.sleep(.1*np.random.rand())
        imagelist = self.im.get_imagelist(exp, name_database=name_database)
        locked = (imagelist=='locked')

        # 2- Doing the edge extraction for each image in this list
        if not(locked):
            try:
                edgeslist = np.load(matname + '_edges.npy')
            except Exception, e:
                log.info(' >> There is no edgeslist: %s ', e)
                log.info('>> Doing the edge extraction')
                edgeslist = self.full_run(exp, name_database, imagelist, noise=noise)
                if edgeslist == 'locked':
                    log.info('>> Edge extraction %s is locked', matname)
                    locked = True
                else:
                    np.save(matname + '_edges.npy', edgeslist)
        else: return 'locked', 'locked imagelist'

        # 3- Doing the independence check for this set
        if not(locked):
            txtname = os.path.join(self.pe.figpath, exp + '_dependence_' + name_database + note + '.txt')
            if not(os.path.isfile(txtname)) and not(os.path.isfile(txtname + '_lock')):
                file(txtname + '_lock', 'w').close() # touching
                log.info(' >> Doing check_independence')
                out = self.check_independence(self.cohistedges(edgeslist, name_database, symmetry=False, display=None), name_database)
                f = file(txtname, 'w')
                f.write(out)
                f.close()
                out = self.check_independence(self.cohistedges(edgeslist, name_database, symmetry=True, display=None), name_database)
                f = file(os.path.join(self.pe.figpath, exp + '_dependence_sym_' + name_database + note + '.txt'), 'w')
                f.write(out)
                f.close()
                print out
                try:
                    os.remove(txtname + '_lock')
                except Exception, e:
                    log.error('Failed to remove lock file %s_lock, error : %s ', txtname, e)

        # 4- Doing the edge figures to check the edge extraction process
        edgedir = os.path.join(self.pe.edgefigpath, exp + '_' + name_database)
        if not(os.path.isdir(edgedir)): os.mkdir(edgedir)

        if not(locked):
            N_image = edgeslist.shape[2]
            for i_image in range(N_image):
                filename, croparea = imagelist[i_image]

                figname = os.path.join(edgedir, filename.replace('.png', '') + str(croparea) + '.png')
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    try:
                        file(figname + '_lock', 'w').close()
                        log.info(' redoing figure %s ', figname)
                        image, filename_, croparea_ = self.im.patch(name_database=name_database, filename=filename, croparea=croparea)
                        if noise >0.: image += noise*image[:].std()*np.random.randn(image.shape[0], image.shape[1])
#                        if self.do_whitening: image = self.im.whitening(image)
                        fig, a = self.show_edges(edgeslist[:, :, i_image], image=image*1.)
                        fig.savefig(figname)
                        plt.close(fig)
                        try:
                            os.remove(figname + '_lock')
                        except Exception, e:
                            log.info('Failed to remove lock file %s_lock', ', error : %s ', figname , e)
                    except Exception, e:
                        log.info('Failed to make edge image  %s, error : %s ', figname , e)

                figname = os.path.join(edgedir, filename.replace('.png', '') + str(croparea) + '_reconstruct.png')
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    try:
                        file(figname + '_lock', 'w').close()
                        log.info(' reconstructing figure %s ', figname)
                        image, filename_, croparea_  = self.im.patch(name_database=name_database, filename=filename, croparea=croparea)
                        if self.do_whitening: image = self.im.whitening(image)
                        image_ = self.reconstruct(edgeslist[:, :, i_image])
                        #if self.do_whitening: image_ = self.im.dewhitening(image_)
                        fig, a = self.show_edges(edgeslist[:, :, i_image], image=image_*1.)
                        fig.savefig(figname)
                        plt.close(fig)
                        try:
                            os.remove(figname + '_lock')
                        except Exception, e:
                            log.error('Failed to remove lock file %s_lock, error : %s ', figname, e)
                    except Exception, e:
                        log.error('Failed to make reconstruct image  %s , error : %s  ', figname, e)

            # 5- Computing RMSE to check the edge extraction process
            try:
                RMSE = np.load(matname + '_RMSE.npy')
            except Exception, e:
                log.info(' >> There is no RMSE: %s ', e)
                if not(os.path.isfile(matname + '_RMSE.npy_lock')):
                    file(matname + '_RMSE.npy_lock', 'w').close()
                    N_image = edgeslist.shape[2]
                    RMSE = np.zeros((N_image,))
                    for i_image in range(N_image):
                        filename, croparea = imagelist[i_image]
                        image, filename_, croparea_  = self.im.patch(name_database=name_database, filename=filename, croparea=croparea)
                        if self.do_whitening: image = self.im.whitening(image)
                        image_ = self.reconstruct(edgeslist[:, :, i_image])
#                        print image.mean(), image.std(), image_.mean(), image_.std()
                        X, Y = np.mgrid[-1:1:1j*self.N_X, -1:1:1j*self.N_Y]
                        mask = (X**2 + Y**2) < 1.
                        RMSE[i_image] =  ((image*mask-image_*mask)**2).sum()/((image*mask)**2).sum()
    #                    print 'RMSE = ', RMSE[i_image]
                    np.save(matname + '_RMSE.npy', RMSE)
                    try:
                        os.remove(matname + '_RMSE.npy_lock')
                    except Exception, e:
                        log.error('Failed to remove lock file %s_RMSE.npy_lock, error : %s ', matname, e)
                else:
                    log.warn(' Some process is building the RMSE: %s_RMSE.npy', matname)

            if not(os.path.isfile(matname + '_RMSE.npy_lock')):
                log.info('>>> For the class %s, in experiment %s RMSE = %f ', name_database, exp, RMSE.mean())


            # 6- Plotting the histogram

#            figname = os.path.join(self.pe.figpath, exp + '_proba-scale_' + name_database + note + self.pe.ext)
#            if not(os.path.isfile(figname)):
#                fig, a = self.histedges_scale(edgeslist, display=True)
#                fig.savefig(figname)
#                plt.close(fig)
#
            figname = os.path.join(self.pe.figpath, exp + '_proba-theta_' + name_database + note + self.pe.ext)
            if not(os.path.isfile(figname)):
                fig, a = self.histedges_theta(edgeslist, display=True)
                fig.savefig(figname)
                plt.close(fig)

#            figname = os.path.join(self.pe.figpath, exp + '_proba-cohist_scale_' + name_database + note + self.pe.ext)
#            if not(os.path.isfile(figname)):
#                fig, a = self.cohistedges(edgeslist, display='cohist_scale')
#                fig.savefig(figname)
#                plt.close(fig)
#
            figname = os.path.join(self.pe.figpath, exp + '_proba-edgefield_colin_' + name_database + note + self.pe.ext)
            if not(os.path.isfile(figname)):
                fig, a = self.cohistedges(edgeslist, symmetry=False, display='colin_geisler')
                fig.savefig(figname)
                plt.close(fig)

            figname = os.path.join(self.pe.figpath, exp + '_proba-edgefield_cocir_' + name_database + note + self.pe.ext)
            if not(os.path.isfile(figname)):
                fig, a = self.cohistedges(edgeslist, symmetry=False, display='cocir_geisler')
                fig.savefig(figname)
                plt.close(fig)

            figname = os.path.join(self.pe.figpath, exp + '_proba-edgefield_chevrons_' + name_database + note + self.pe.ext)
            if not(os.path.isfile(figname)):
                fig, a = self.cohistedges(edgeslist, display='chevrons')
                fig.savefig(figname)
                plt.close(fig)

            return imagelist, edgeslist
        else:
            return 'locked', 'locked edgeslist'

    # some helper funtion to compare the databases
    def KL(self, v_hist, v_hist_obs):
        if v_hist.sum()==0 or v_hist_obs.sum()==0: log.error('>X>X>X KL function:  problem with null histograms! <X<X<X<')
        if True:
            v_hist /= v_hist.sum()
            v_hist_obs /= v_hist_obs.sum()
            # taking advantage of log(True) = 0 and canceling out null bins in v_hist_obs
            return np.sum(v_hist.ravel()*(np.log(v_hist.ravel()+(v_hist == 0).ravel()) - np.log(v_hist_obs.ravel()+(v_hist_obs == 0).ravel())))
        else:
            from scipy.stats import entropy
            return entropy(v_hist_obs, v_hist, base=2)

    def check_independence(self, v_hist, name_database, labels=['d', 'phi', 'theta', 'scale']):
        v_hist /= v_hist.sum()
        fullset = [0, 1, 2, 3]
#    from scipy.stats import entropy
#    print KL(v_hist, v_hist),  entropy(v_hist.ravel(), v_hist.ravel())
        flat = np.ones_like(v_hist)
        flat /= flat.sum()
        out = 'Checking dependence in ' + name_database + '\n'
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
    print 'main'
#     from plt import imread
#     # whitening
#     image = imread('database/gris512.png')[:,:,0]
#     lg = LogGabor(image.shape)
