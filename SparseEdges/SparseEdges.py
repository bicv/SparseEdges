"""
SparseEdges

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

    def run_mp(self, image, verbose=False, progress=False, progressfile=None):
        """
        runs the MatchingPursuit algorithm on image

        """
        edges = np.zeros((6, self.pe.N))
        C = self.linear_pyramid(image) # check LogGabor package
        if progressfile is not None:
            t0 = time.time()
        if progress:
            import pyprind
            my_prbar = pyprind.ProgPercent(self.pe.N)   # 1) initialization with number of iterations
        for i_edge in range(self.pe.N):
            # MATCHING
            ind_edge_star = self.argmax(C) # check LogGabor package
            coeff = self.pe.MP_alpha * np.abs(C[ind_edge_star])
            # recording
            edges[:, i_edge] = np.array([ind_edge_star[0]*1., ind_edge_star[1]*1.,
                                         self.theta[ind_edge_star[2]],
                                         self.sf_0[ind_edge_star[3]],
                                         coeff, np.angle(C[ind_edge_star])])
            # PURSUIT
            C = self.backprop(C, ind_edge_star)
            # reporting
            if verbose: print('Edge ', i_edge, '/', self.pe.N, ' - Max activity  : ', '%.3f' % np.absolute(C[ind_edge_star]), ' phase= ', '%.3f' % np.angle(C[ind_edge_star], deg=True), ' deg,  @ ', ind_edge_star)
            if progress: my_prbar.update()
            if progressfile is not None:
                #if i_edge / (self.pe.N/32)
                f = open(progressfile, 'w')
                f.write('Edge ' + str(i_edge) + '/' + str(self.pe.N) + 'in %.3f' % (time.time() -t0) + 's - ' + progressfile + '\n')
                f.close()

        return edges, C

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
        if do_mask: lg_star *= self.mask
        for i_sf_0, sf_0 in enumerate(self.sf_0):
            for i_theta, theta in enumerate(self.theta):
                FT_lg = self.loggabor(0., 0., sf_0=sf_0, B_sf=self.pe.B_sf, theta=theta, B_theta=self.pe.B_theta)
                C[:, :, i_theta, i_sf_0] -= self.FTfilter(lg_star, FT_lg, full=True)
        return C

    def reconstruct(self, edges, do_mask=False, do_energy=False):
        image = np.zeros((self.pe.N_X, self.pe.N_Y))
        for i_edge in range(edges.shape[1]):
            image += self.invert(edges[4, i_edge] * np.exp(1j*edges[5, i_edge]) *
                            self.loggabor(
                                            edges[0, i_edge], edges[1, i_edge],
                                            theta=edges[2, i_edge], B_theta=self.pe.B_theta,
                                            sf_0=edges[3, i_edge], B_sf=self.pe.B_sf,
                                            ),
                            full=False)
        if do_mask: image *= self.mask
        return image

    def show_edges(self, edges, fig=None, ax=None, image=None, norm=True,
                   color='auto', v_min=-1., v_max=1., show_phase=True, gamma=1.,
                   pedestal=0., show_mask=False, mappable=False, scale=None):
        """
        Shows the quiver plot of a set of edges, optionally associated to an image.

        """
        import matplotlib.cm as cm
        if fig==None:
            #  Figure :                      height         ----------           width
            # figsize = w, h tuple in inches
            fig = plt.figure(figsize=(self.pe.figsize*self.pe.N_Y/self.pe.N_X, self.pe.figsize))
        if ax==None:
            border = 0.0
            ax = fig.add_axes((border, border, 1.-2*border, 1.-2*border), facecolor='w')
        ax.axis(c='b', lw=0, frame_on=False)

        if color in ['black', 'redgreen', 'redblue', 'bluegreen', 'brown', 'green', 'blue']: #cocir or chevrons
            linewidth = self.pe.line_width_chevrons
            if scale is None: scale = self.pe.scale_chevrons
        else:
            linewidth = self.pe.line_width
            if scale is None: scale = self.pe.scale

        opts= {'extent': (0, self.pe.N_Y, self.pe.N_X, 0), # None, #
               'cmap': cm.gray,
               'vmin':v_min, 'vmax':v_max, 'interpolation':'nearest'}#, 'origin':'upper'}
#         origin : [‘upper’ | ‘lower’], optional, default: None
#         Place the [0,0] index of the array in the upper left or lower left corner of the axes. If None, default to rc image.origin.
#         extent : scalars (left, right, bottom, top), optional, default: None
#         Data limits for the axes. The default assigns zero-based row, column indices to the x, y centers of the pixels.
        if type(image)==np.ndarray:
#             if image.ndim==2: opts['cmap'] = cm.gray
            if norm: image = self.normalize(image, center=True, use_max=True)
            ax.imshow(image, **opts)
        else:
            ax.imshow([[v_max]], **opts)
        if edges.shape[1] > 0:
            from matplotlib.collections import LineCollection, PatchCollection
            import matplotlib.patches as patches
            # draw the segments
            segments, colors, linewidths = list(), list(), list()
            patch_circles = []

            Y, X, Theta, Sf_0 = edges[0, :]+.5, edges[1, :]+.5, np.pi -  edges[2, :], edges[3, :]

            weights = edges[4, :]
            weights = weights/(np.abs(weights)).max()
            phases = edges[5, :]

            for x, y, theta, sf_0, weight, phase in zip(X, Y, Theta, Sf_0, weights, phases):
                #if (not show_mask) or ((y/self.pe.N_X -.5)**2+(x/self.pe.N_Y -.5)**2) < .5**2:
                u_, v_ = np.cos(theta)*scale/sf_0, np.sin(theta)*scale/sf_0
                segment = [(x - u_, y - v_), (x + u_, y + v_)]
                segments.append(segment)
                if color=='auto':
                    if not(show_phase):
                        fc = cm.hsv(0, alpha=pedestal + (1. - pedestal)*weight**gamma)
                    else:
                        fc = cm.hsv((phase/np.pi/2) % 1., alpha=pedestal + (1. - pedestal)*weight**gamma)
                        # check-out  https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/

                elif color == 'black':
                    fc = (0, 0, 0, 1)# black
                elif color == 'green': # figure 1DE
                    fc = (0.05, 0.5, 0.05, np.abs(weight)**gamma)
                elif color == 'redgreen': # figure 1DE
                    fc = (0.5, 0.5, 0.05, np.abs(weight)**gamma)
                elif color == 'redblue': # figure 1DE
                    fc = (0.5, 0.05, 0.5, np.abs(weight)**gamma)
                elif color == 'blue': # figure 1DE
                    fc = (0.05, 0.05, 0.5, np.abs(weight)**gamma)
                elif color == 'brown': # figure 1DE
                    fc = (0.5, 0.05, 0.05, np.abs(weight)**gamma)
                else:
                    fc = ((np.sign(weight)+1)/2, 0, (1-np.sign(weight))/2, np.abs(weight)**gamma)
                colors.append(fc)
                linewidths.append(linewidth) # *weight thinning byalphax...
                patch_circles.append(patches.Circle((x, y), self.pe.scale_circle*scale/sf_0, lw=0., facecolor=fc, edgecolor='none'))

            line_segments = LineCollection(segments, linewidths=linewidths, colors=colors, linestyles='solid')
            ax.add_collection(line_segments)
            circles = PatchCollection(patch_circles, match_original=True)
            ax.add_collection(circles)

        plt.setp(ax, xticks=[])
        plt.setp(ax, yticks=[])

        if show_mask:
            linewidth_mask = 1 #
            from matplotlib.patches import Ellipse
            circ = Ellipse((.5*self.pe.N_Y, .5*self.pe.N_X),
                            self.pe.N_Y-linewidth_mask, self.pe.N_X-linewidth_mask,
                            fill=False, facecolor='none', edgecolor = 'black', alpha = 0.5, ls='dashed', lw=linewidth_mask)
            ax.add_patch(circ)
        ax.axis([0, self.pe.N_Y, self.pe.N_X, 0])
        ax.grid(b=False, which="both")
        plt.draw()
        if mappable:
            return fig, ax, line_segments
        else:
            return fig, ax

    def full_run(self, exp, name_database, imagelist, noise, N_do=2, time_sleep=.1):
        """
        runs the edge extraction for a list of images

        """
        #self.mkdir()
        for path in self.pe.matpath, self.pe.edgematpath:
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
                        open(matname + '_lock', 'w').close()
                        self.log.info('Doing edge extraction of %s ', matname)
                        image, filename_, croparea_ = self.patch(name_database, filename=filename, croparea=croparea, do_whitening=self.pe.do_whitening)
                        if noise > 0.: image += noise*image[:].std()*self.texture(filename=filename, croparea=croparea)
                        edges, C = self.run_mp(image, verbose=self.pe.verbose>50, progressfile=matname + '_lock')
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
                for i_image, (filename, croparea) in enumerate(imagelist):
                    matname = os.path.join(self.pe.edgematpath, exp + '_' + name_database, filename + str(croparea) + '.npy')
                    edgeslist[:, :, i_image] = np.load(matname)
                return edgeslist
            except Exception as e:
                self.log.error(' some locked edge extractions %s, error on file %s', e, matname)
                return 'locked'

    def full_MSE(self, exp, name_database, imagelist):
        matname = os.path.join(self.pe.edgematpath, exp + '_' + name_database)
        #self.mkdir()
        if not(os.path.isdir(matname)): os.mkdir(matname)
        edgeslist = np.load(matname + '_edges.npy')
        N_do = 2
        for _ in range(N_do): # repeat this loop to make sure to scan everything
            global_lock = False # will switch to True when we resume a batch and detect that one edgelist is not finished in another process
            for i_image, (filename, croparea) in enumerate(imagelist):
                matname = os.path.join(self.pe.edgematpath, exp + '_' + name_database, filename + str(croparea) + '_MSE.npy')
                if not(os.path.isfile(matname)):
                    if not(os.path.isfile(matname + '_lock')):
                        open(matname + '_lock', 'w').close()
                        # loading image
                        image, filename_, croparea_ = self.patch(name_database, filename=filename, croparea=croparea, do_whitening=self.pe.do_whitening)
                        # loading edges
                        edges = edgeslist[:, :, i_image]
                        # computing MSE
                        MSE = np.ones(self.pe.N)
                        image_rec = np.zeros_like(image)
                        for i_N in range(self.pe.N):
                            MSE[i_N] =  ((image-image_rec)**2).sum()
                            image_rec += self.reconstruct(edges[:, i_N][:, None], do_mask=False)
                        np.save(matname, MSE)
                        try:
                            os.remove(matname + '_lock')
                        except Exception as e:
                            self.log.error('Failed to remove lock file %s_lock, error : %s ', matname, traceback.print_tb(sys.exc_info()[2]))
                    else:
                        self.log.info('The edge extraction at step %s is locked', matname)
                        global_lock = True
        if global_lock is True:
            self.log.error(' some locked MSE extractions ')
            return 'locked'
        else:
            try:
                N_image = len(imagelist)
                MSE = np.ones((N_image, self.pe.N))
                for i_image in range(N_image):
                    filename, croparea = imagelist[i_image]
                    matname_MSE = os.path.join(self.pe.edgematpath, exp + '_' + name_database, filename + str(croparea) + '_MSE.npy')
                    MSE[i_image, :] = np.load(matname_MSE)
                return MSE
            except Exception as e:
                self.log.error(' some locked MSE extractions %s, error ', e)
                return 'locked'


    def texture(self, N_edge=256, a=None, filename='', croparea='', randn=True):
        # a way to get always the same seed for each image
        if not (filename==''):# or not (croparea==''):
            import hashlib
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
                from scipy.stats import powerlaw

                edgeslist[3, :] =  self.sf_0.max() * powerlaw.rvs(a=4., size = N_edge)
                edgeslist[4, :]  = np.random.pareto(a=a, size=(N_edge)) + 1


            edgeslist[5, :] = 2*np.pi*np.random.rand(N_edge)
            image_rec = self.reconstruct(edgeslist, do_mask=False)
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


    def init_binedges(self, mp_theta=None):
        # configuring histograms
        # sequence of scalars,it defines the bin edges, including the rightmost edge.
        self.binedges_d = np.linspace(self.pe.d_min, self.pe.d_max, self.pe.N_r+1)
        self.binedges_phi = np.linspace(-np.pi/2, np.pi/2, self.pe.N_phi+1) + np.pi/self.pe.N_phi/2
        if mp_theta is None:
            mp_theta = np.linspace(-np.pi/2, np.pi/2, self.pe.N_Dtheta+1)[1:]
        theta_bin = (mp_theta + np.hstack((mp_theta[-1]-np.pi, mp_theta[:-1]))) /2
        self.binedges_theta = np.hstack((theta_bin, theta_bin[0]+np.pi))
        # self.binedges_sf_0 = 2**np.arange(np.ceil(np.log2(self.pe.N_X)))
        self.binedges_sf_0 = 1. / np.logspace(.5, self.n_levels+.5, self.n_levels+1, base=self.pe.base_levels)
        self.binedges_sf_0 = self.binedges_sf_0[::-1]
        self.binedges_loglevel = np.linspace(-self.pe.loglevel_max, self.pe.loglevel_max, self.pe.N_scale+1)

    def histedges_theta(self, edgeslist, mp_theta=None, v_hist=None, fig=None, ax=None, figsize=None, display=True, mode='full'):
        """
        First-order stats

        p(theta | I )

        """
        self.init_binedges(mp_theta)

        if mode=='edge':
            return edgeslist[2, :]

        if v_hist is None:
            theta = edgeslist[2, ...].ravel()
            value = edgeslist[4, ...].ravel()

            if self.pe.edge_mask:
                # remove edges whose center position is not on the central disk
                x , y = edgeslist[0, ...].ravel().real, edgeslist[1, ...].ravel().real
                mask = ((y/self.pe.N_X -.5)**2+(x/self.pe.N_Y -.5)**2) < .5**2
                theta = theta[mask]
                value = value[mask]

            weights = np.absolute(value)/(np.absolute(value)).sum()
            v_hist, v_theta_edges = np.histogram(theta, bins=self.binedges_theta, density=True, weights=weights)

        if display:
            # v_hist /= v_hist.sum()
            if figsize is None: figsize = (self.pe.figsize_hist, self.pe.figsize_hist)
            if fig is None: fig = plt.figure(figsize=figsize)
            if ax is None: ax = plt.axes(polar=True, facecolor='w')
            width = self.binedges_theta[1:] - self.binedges_theta[:-1]
            # in polar coordinates, probability should be made proportional to
            # the surface, i.e. bars height to the square root of probability
            # see https://laurentperrinet.github.io/sciblog/posts/2014-12-09-polar-bar-plots.html
            ax.bar(self.binedges_theta[:-1], (v_hist)**.5, width=width, color='#66c0b7', align='edge')# edgecolor="none")

            ax.bar(self.binedges_theta[:-1]+np.pi, (v_hist)**.5, width=width, color='#32ab9f', align='edge')
            ax.plot(self.binedges_theta, np.ones_like(self.binedges_theta)*np.sqrt(v_hist.mean()), 'r--')
            ax.plot(self.binedges_theta+np.pi, np.ones_like(self.binedges_theta)*np.sqrt(v_hist.mean()), 'r--')
            plt.setp(ax, yticks=[])
            return fig, ax
        else:
            return v_hist, self.binedges_theta

    def histedges_scale(self, edgeslist, fig=None, ax=None, display=True):
        """
        First-order stats for the scale

        p(scale | I )

        """
        self.init_binedges()

        sf_0 = (edgeslist[3, ...].ravel())
        value = edgeslist[4, ...].ravel()
        if self.pe.edge_mask:
            # remove edges whose center position is not on the central disk
            x , y = edgeslist[0, ...].ravel().real, edgeslist[1, ...].ravel().real
            mask = ((y/self.pe.N_X -.5)**2+(x/self.pe.N_Y -.5)**2) < .5**2
            sf_0 = sf_0[mask]
            value = value[mask]
        weights = np.absolute(value)/(np.absolute(value)).sum()
        v_hist, v_sf_0_edges_ = np.histogram(sf_0, self.binedges_sf_0, density=False, weights=weights)
        v_hist /= v_hist.sum()
        if display:
            if fig==None: fig = plt.figure(figsize=(self.pe.figsize_hist, self.pe.figsize_hist))
            if ax==None: ax = fig.add_subplot(111, facecolor='w')
            ax.step(v_sf_0_edges_[:-1], v_hist, c='k')
            #plt.setp(ax, yticks=[])
            plt.xlabel(r'$sf_0$')
            plt.ylabel('probability')
            return fig, ax
        else:
            return v_hist, v_sf_0_edges_


    def cooccurence(self, edges_ref, edges_comp=None, symmetry=True):
        if edges_comp is None: edges_comp = edges_ref.copy()
        if self.pe.edge_mask:
            # remove edges whose center position is not on the central disk
            mask = ((edges_ref[1, :]/self.pe.N_X -.5)**2+(edges_ref[0, :]/self.pe.N_Y -.5)**2) < .5**2
            edges_ref = edges_ref[:, mask]
        # retrieve individual positions, orientations, scales and coefficients
        X_ref, Y_ref = edges_ref[0, :], edges_ref[1, :]
        Theta_ref = edges_ref[2, :]
        Sf_0_ref = edges_ref[3, :]
        value_ref = edges_ref[4, :]
        phase_ref = edges_ref[5, :]
        X_comp, Y_comp = edges_comp[0, :], edges_comp[1, :]
        Theta_comp = edges_comp[2, :]
        Sf_0_comp = edges_comp[3, :]
        value_comp = edges_comp[4, :]
        phase_comp = edges_comp[5, :]

        # TODO: include phases or use that to modify center of the edge
        # TODO: or at least on the value (ON or OFF) of the edge
        # TODO: normalize weights by their relative order to be independent of the texture
        # TODO : make an histogram on log-radial coordinates and theta versus scale
        # TODO: check that we correctly normalize position by the scale of the current edge

        # to control if we raise an error on numerical error, we use
        np.seterr(all='ignore')
        dx = X_ref[:, None] - X_comp[None, :]
        dy = Y_ref[:, None] - Y_comp[None, :]
        d = np.sqrt(dx**2 + dy**2) / self.pe.N_X  # distance normalized by the image size (H)
        if self.pe.scale_invariant:
            d /= np.sqrt(np.median(Sf_0_ref)*np.median(Sf_0_comp))
            d *= np.sqrt(Sf_0_ref[:, None]*Sf_0_comp[None, :])
        d *= self.pe.d_width # distance in visual angle
        theta = Theta_ref[:, None] - Theta_comp[None, :]
        phi = np.arctan2(dy, dx) - np.pi/2 - Theta_comp[None, :]
        if symmetry: phi -= theta/2
        loglevel = np.log2(Sf_0_ref[:, None]) - np.log2(Sf_0_comp[None, :])
        # putting everything in the right range:
        phi = ((phi + np.pi/2  - np.pi/self.pe.N_phi/2 ) % (np.pi)) - np.pi/2  + np.pi/self.pe.N_phi/2
        theta = ((theta + np.pi/2 - np.pi/self.pe.n_theta/2)  % (np.pi) ) - np.pi/2  + np.pi/self.pe.n_theta/2
        dphase = phase_ref[:, None] - phase_comp[None, :]
        logvalue = np.log2(value_ref[:, None]) - np.log2(value_comp[None, :])
        return d, phi, theta, loglevel, dphase, logvalue

    def cooccurence_hist(self, edges_ref, edges_comp=None, symmetry=True, mode='full'):
        if edges_comp is None: edges_comp = edges_ref.copy()

        if mode=='edge':
            N_edge = edges_ref.shape[1]
            v_hist = np.zeros((self.pe.N_r, self.pe.N_phi, self.pe.N_Dtheta, self.pe.N_scale, N_edge))
            for i_edge in range(N_edge):
                v_hist[..., i_edge] = self.cooccurence_hist(edges_ref=edges_ref[:, i_edge][:, None], edges_comp=edges_comp, symmetry=symmetry, mode='full')
            return v_hist
        else:
            d, phi, theta, loglevel, dphase, logvalue = self.cooccurence(edges_ref, edges_comp, symmetry=symmetry)

            #computing weights
            # normalize weights by the max (while some images are "weak")? the corr coeff would be an alternate solution... / or simply the rank
            Weights = edges_ref[4, :]
            if self.pe.do_rank: Weights[Weights.argsort()] = np.linspace(1./Weights.size, 1., Weights.size)
            weights = Weights[:, None] * edges_comp[4, :][None, :]
            if self.pe.weight_by_distance:
                # normalize weights by the relative distance (bin areas increase with radius)
                # it makes sense to give less weight to "far bins"
                weights /= (d + 1.e-6) # warning, some are still at the same position d=0...
                # exclude self-occurence
                # weights[d==0] = 0.
                weights[d<self.pe.d_min] = 0.
            if not self.pe.multiscale:
                # selecting only co-occurrences at the same scale
                weights *= (edges_ref[3, :][:, None]==edges_comp[3, :][None, :])
            # just checking if we get different results when selecting edges with a similar phase (von Mises profile)
            if self.pe.kappa_phase>0:
                # TODO: should we use the phase information to refine position?
                # https://en.wikipedia.org/wiki/Atan2
                weights *= np.exp(self.pe.kappa_phase*np.cos(np.arctan2(dphase)))
                # there may be a locus depending on dx and value

            weights /= weights.sum()

            #computing histogram
            self.init_binedges()
            v_hist_, edges_ = np.histogramdd([d.ravel(), phi.ravel(), theta.ravel(), loglevel.ravel()], #data,
                                             bins=(self.binedges_d, self.binedges_phi, self.binedges_theta, self.binedges_loglevel),
                                             normed=True, # TODO check if correct True,
                                             weights = weights.ravel()
                                            )
            if v_hist_.sum()<.01: self.log.debug(' less than 1 percent of co-occurences within ranges: %f ', v_hist_.sum())

            return v_hist_

    def cohistedges(self, edgeslist, v_hist=None, prior=None,
                    fig=None, ax=None, symmetry=True,
                    display='chevrons', v_min=None, v_max=None, labels=True, mappable=False, radius=None,
                    xticks=False, half=False, dolog=True, color='redblue', colorbar=True, cbar_label=True):
        """
        second-order stats= center all edges around the current one by rotating and scaling

        p(x-x_, y-y_, theta-theta_ | I, x_, y_, theta_)

        """
        self.init_binedges()

        if not(edgeslist is None):
            v_hist = None
            six, N_edge, N_image = edgeslist.shape
            for i_image in range(N_image):
                v_hist_ = self.cooccurence_hist(edgeslist[:, :, i_image], symmetry=symmetry)
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
            a1 = fig.add_subplot(221, facecolor='w')#, polar = True)
            a1.imshow((v_hist.sum(axis=3).sum(axis=2)), **options)
            if symmetry:
                a1.set_xlabel('psi')
            else:
                a1.set_xlabel('phi')
            a1.set_xticks([0., self.binedges_phi.size/2. -1.5, self.binedges_phi.size-2.])
            a1.set_xticklabels(['-pi/2 + bw ', '0', 'pi/2'])
            a1.set_ylabel('d')
            edges_d_half = .5*(self.binedges_d[1:] + self.binedges_d[:-1])
            a1.set_yticks([0., self.binedges_d.size-2.])
            a1.set_yticklabels([str(edges_d_half[0]), str(edges_d_half[-1])])
            a1.axis('tight')
            a2 = fig.add_subplot(222, facecolor='w')#edges_[0], edges_[2],
            a2.imshow((v_hist.sum(axis=3).sum(axis=1)), **options)
            a2.set_xlabel('theta')
            a2.set_xticks([0., self.binedges_theta.size/2.-1.5, self.binedges_theta.size-2.])
            a2.set_xticklabels(['-pi/2 + bw', '0', 'pi/2'])
            a2.set_ylabel('d')
            a2.set_yticks([0., self.binedges_d.size-2.])
            a2.set_yticklabels([str(edges_d_half[0]), str(edges_d_half[-1])])
            a2.axis('tight')
            a3 = fig.add_subplot(223, facecolor='w')#edges_[1], edges_[2],
            a3.imshow((v_hist.sum(axis=3).sum(axis=0)).T, **options)
            if symmetry:
                a3.set_xlabel('psi')
            else:
                a3.set_xlabel('phi')
            a3.set_xticks([0., self.binedges_phi.size/2. - 1.5, self.binedges_phi.size-2.])
            a3.set_xticklabels(['-pi/2 + bw', '0', 'pi/2'])
            a3.set_ylabel('theta')
            a3.set_yticks([0., self.binedges_theta.size/2. - 1.5, self.binedges_theta.size-2.])
            a3.set_yticklabels(['-pi/2 + bw', '0', 'pi/2'])
            a3.axis('tight')
            a4 = fig.add_subplot(224, facecolor='w')#, polar = True)
            a4.imshow((v_hist.sum(axis=1).sum(axis=1)), **options)
            a4.set_xlabel('levels')
            a4.set_xticks([0., self.pe.N_scale/2. -.5, self.pe.N_scale -1.])
            a4.set_xticklabels(['smaller', '0', 'bigger'])
            a4.set_ylabel('d')
            a4.set_yticks([0., self.binedges_d.size-2.])
            a4.set_yticklabels([str(edges_d_half[0]), str(edges_d_half[-1])])
            a4.axis('tight')
#             plt.tight_layout()
            return fig, (a1, a2, a3, a4)

        elif display=='colin_geisler':
            edge_scale = 8.
            try:
                if fig==None:
                    fig = plt.figure(figsize=(self.pe.figsize_cohist, self.pe.figsize_cohist))
                    if ax==None:
                        ax = fig.add_subplot(111)
                v_hist_noscale = v_hist.sum(axis=3)
                colin_edgelist = np.zeros((6, self.pe.N_r * self.pe.N_phi * 2 + 1 ))
                colin_argmax = np.argmax(v_hist_noscale, axis=2)
                for i_r, d_r in enumerate(self.binedges_d[:-1]):
                    for i_phi, phi in enumerate(self.binedges_phi[:-1]):
                        rad = d_r / self.pe.d_max * max(self.pe.N_X, self.pe.N_Y) /2
                        ii_phi = i_r * self.pe.N_phi
                        colin_edgelist[0:2, ii_phi + i_phi] =  self.pe.N_X /2 - rad * np.sin(phi + np.pi/self.pe.N_phi/2), self.pe.N_Y /2 + rad * np.cos(phi + np.pi/self.pe.N_phi/2)
                        colin_edgelist[2, ii_phi + i_phi] = self.binedges_theta[colin_argmax[i_r, i_phi]] + np.pi/self.pe.N_Dtheta/2
                        colin_edgelist[3, ii_phi + i_phi] = edge_scale
                        colin_edgelist[4, ii_phi + i_phi] = v_hist_noscale[i_r, i_phi, colin_argmax[i_r, i_phi]]
                        # symmetric
                        colin_edgelist[:, ii_phi + i_phi +  self.pe.N_r * self.pe.N_phi] = colin_edgelist[:, ii_phi + i_phi]
                        colin_edgelist[0:2, ii_phi + i_phi +  self.pe.N_r * self.pe.N_phi] = self.pe.N_X - colin_edgelist[0, ii_phi + i_phi], self.pe.N_Y - colin_edgelist[1, ii_phi + i_phi]
                # reference angle
                colin_edgelist[:, -1] = [self.pe.N_X /2, self.pe.N_Y /2, 0, edge_scale, colin_edgelist[4,:].max() *1.2, 0.]
                return self.show_edges(colin_edgelist, fig=fig, ax=ax, image=None, v_min=0., v_max=v_hist_noscale.max(), color=color, scale=40.)
            except Exception as e:
                self.log.error(' failed to generate colin_geisler plot, %s', traceback.print_tb(sys.exc_info()[2]))
                return e, None # used to return something instead of None

        elif display=='cocir_geisler':
            edge_scale = 8.
            try:
                if fig==None:
                    fig = plt.figure(figsize=(self.pe.figsize_cohist, self.pe.figsize_cohist))
                    if ax==None:
                        ax = fig.add_subplot(111)
                v_hist_noscale = v_hist.sum(axis=3)
                cocir_edgelist = np.zeros((6, self.pe.N_r * self.pe.N_Dtheta * 2 + 1 ))
                cocir_proba = np.argmax(v_hist_noscale, axis=1)
                for i_r, d_r in enumerate(self.binedges_d[:-1]):
                    for i_theta, theta in enumerate(self.binedges_theta[:-1]):
                        rad = d_r / self.pe.d_max * max(self.pe.N_X, self.pe.N_Y) /2
                        ii_theta = i_r * self.pe.N_Dtheta
                        cocir_edgelist[0:2, ii_theta + i_theta] =  self.pe.N_X /2 - rad * np.sin( self.binedges_phi[cocir_proba[i_r, i_theta]] + np.pi/self.pe.N_phi/2), self.pe.N_Y /2 + rad * np.cos( self.binedges_phi[cocir_proba[i_r, i_theta]] + np.pi/self.pe.N_phi/2)
                        cocir_edgelist[2, ii_theta + i_theta] = theta + np.pi/self.pe.N_Dtheta/2
                        cocir_edgelist[3, ii_theta + i_theta] = edge_scale
                        cocir_edgelist[4, ii_theta + i_theta] = v_hist_noscale[i_r, cocir_proba[i_r, i_theta], i_theta]
                        # symmetric
                        cocir_edgelist[:, ii_theta + i_theta +  self.pe.N_r * self.pe.N_Dtheta] = cocir_edgelist[:,  ii_theta + i_theta]
                        cocir_edgelist[0:2, ii_theta + i_theta +  self.pe.N_r * self.pe.N_Dtheta] = self.pe.N_X - cocir_edgelist[0,  ii_theta + i_theta], self.pe.N_Y - cocir_edgelist[1, ii_theta + i_theta]
                cocir_edgelist[:, -1] = [self.pe.N_X /2, self.pe.N_Y /2, 0, edge_scale, cocir_edgelist[4,:].max() *1.2, 0.]
                return self.show_edges(cocir_edgelist, fig=fig, ax=ax, image=None, v_min=0., v_max=v_hist_noscale.max(), color=color, scale=40.)
            except Exception as e:
                self.log.error(' failed to generate cocir_geisler plot, %s', traceback.print_tb(sys.exc_info()[2]))
                return e, None # used to retrun something instead of None

        elif display=='cohist_scale':
            try:
                if fig==None:
                    fig = plt.figure(figsize=(self.pe.figsize_cohist, self.pe.figsize_cohist))
                    if ax==None:
                        ax = fig.add_subplot(111)
                ax.bar(self.binedges_loglevel[:-1], v_hist.sum(axis=(0, 1, 2)))
                plt.setp(ax, yticks=[])
                ax.set_xlabel('log2 of scale ratio')
                ax.set_ylabel('probability')
                return fig, ax
            except Exception:
                self.log.error(' failed to generate cohist_scale, %s', e)
                return e, None # used to retrun something instead of None


        elif display=='chevrons':
            assert(symmetry==True)
            v_hist_angle = v_hist.sum(axis=0).sum(axis=-1) # -d-,phi,  theta, -scale-
            # some useful normalizations
            if prior is not None:
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
                v_phi, v_theta = self.binedges_phi[(self.pe.N_phi/2-1):-1] + np.pi/self.pe.N_phi/2, self.binedges_theta[(self.pe.N_Dtheta/2-1):-1] + np.pi/self.pe.N_Dtheta/2
                i_phi_shift, i_theta_shift = self.pe.N_phi/2+1, self.pe.N_Dtheta/2-1
            else:
                v_phi, v_theta = self.binedges_phi - np.pi/self.pe.N_phi/2, self.binedges_theta - np.pi/self.pe.N_Dtheta/2
                i_phi_shift, i_theta_shift = 2, -1
            s_phi, s_theta = len(v_phi), len(v_theta)
            #print 'DEBUG: s_phi, s_theta, self.pe.N_phi, self.pe.N_Dtheta', s_phi, s_theta, self.pe.N_phi, self.pe.N_Dtheta
            rad_X, rad_Y = 1.* self.pe.N_X/s_theta, 1.*self.pe.N_Y/s_phi
            rad = min(rad_X, rad_Y) / 2.619
            if radius==None: radius = np.ones((self.pe.N_phi, self.pe.N_Dtheta))

            if fig==None:
                fig = plt.figure(figsize=(self.pe.figsize_cohist, self.pe.figsize_cohist))
                if ax==None:
                    border = 0.005
                    ax = fig.add_axes((border, border, 1.-2*border, 1.-2*border), facecolor='w')
                    ax.axis(c='b', lw=0)

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
            ax.add_collection(p)

#            print rad/s_theta, rad/s_phi
            fig, ax = self.show_edges(angle_edgelist, fig=fig, ax=ax, image=None, color='black')
            ax.axis([0, self.pe.N_Y+1, self.pe.N_X+1, 0])

            if colorbar:
                cbar = plt.colorbar(ax=ax, mappable=p, shrink=0.6)
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

            if labels:
                if not(xticks=='left'): ax.set_xlabel(r'azimuth difference $\psi$')
                if not(xticks=='bottom'): ax.set_ylabel(r'orientation difference $\theta$')
            if not(xticks==False):
                eps = 0.5 # used to center grid.
                if half:
                    plt.setp(ax, xticks=[(1./self.pe.N_phi/1.25)*self.pe.N_X, (1. - 1./self.pe.N_phi/1.25)*self.pe.N_X])
                    if not(xticks=='left'):
                        plt.setp(ax, xticklabels=[r'$0$', r'$\frac{\pi}{2}$'])
                    else:
                        plt.setp(ax, xticklabels=[r'', r''])
                else:
                    plt.setp(ax, xticks=[(1./(self.pe.N_phi+1)/2)*self.pe.N_X+eps, .5*self.pe.N_X+eps, (1. - 1./(self.pe.N_phi+1)/2)*self.pe.N_X+eps])
                    if not(xticks=='left'):
                        plt.setp(ax, xticklabels=[r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$'])
                    else:
                        plt.setp(ax, xticklabels=[r'', r''])
                if half:
                    plt.setp(ax, yticks=[(1./self.pe.N_Dtheta)*self.pe.N_Y, (1. - 1./(self.pe.N_Dtheta+.45))*self.pe.N_Y])
                    if not(xticks=='bottom'):
                        plt.setp(ax, yticklabels=[r'$0$', r'$\frac{\pi}{2}$'])
                    else:
                        plt.setp(ax, yticklabels=[r'', r''])
                else:
                    plt.setp(ax, yticks=[1./(self.pe.N_Dtheta+1)/2*self.pe.N_X+eps, .5*self.pe.N_Y+eps, (1. - 1./(self.pe.N_Dtheta+1)/2)*self.pe.N_Y+eps])
                    if not(xticks=='bottom'):
                        plt.setp(ax, yticklabels=[r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$'])
                    else:
                        plt.setp(ax, yticklabels=['', '', ''])
                plt.grid('off')
            plt.draw()

            return fig, ax
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
        #self.mkdir()
        for path in self.pe.matpath, self.pe.edgematpath:
            if not(os.path.isdir(path)): os.mkdir(path)
        matname = os.path.join(self.pe.edgematpath, exp + '_' + name_database)
        #while os.path.isfile(matname + '_images_lock'):
        imagelist = self.get_imagelist(exp, name_database=name_database)
        locked = (imagelist=='locked')
#         print 'DEBUG: theta used in this experiment: ', self.theta*180/np.pi
        # 2- Doing the edge extraction for each image in this list
        if locked:
            return 'locked imagelist', 'not done', 'not done'
        else:
            try:
                # HACK to regenerate edgeslist
                edgeslist = np.load(matname + '_edges.npy')
            except Exception as e:
                self.log.info(' >> There is no edgeslist: %s ', e)
                time.sleep(.1*np.random.rand())
                edgeslist = self.full_run(exp, name_database, imagelist, noise=noise)
                if edgeslist == 'locked':
                    self.log.info('>> Edge extraction %s is locked', matname)
                    locked = True
                else:
                    np.save(matname + '_edges.npy', edgeslist)

        if locked:
            return imagelist, 'edgeslist not done', 'not started'
        else:
            # Computing MSE to check the edge extraction process
            try:
                MSE = np.load(matname + '_MSE.npy')
            except Exception as e:
                self.log.info(' >> There is no MSE: %s ', e)
                try:
                    MSE = self.full_MSE(exp, name_database, imagelist)
                    if MSE is 'locked':
                        self.log.info('>> MSE extraction %s is locked', matname)
                        locked = True
                    else:
                            np.save(matname + '_MSE.npy', MSE)
                except Exception as e:
                    self.log.error('Failed to compute MSE %s , error : %s ', matname + '_MSE.npy', e)
                    return 'imagelist ok', 'edgelist ok', 'locked MSE'

            try:
                self.log.info('>>> For the class %s, in experiment %s MSE = %f ', name_database, exp, (MSE[:, -1]/MSE[:, 0]).mean())
            except Exception as e:
                locked = True
                self.log.error('Failed to compute average MSE %s ', e)
                return 'imagelist ok', 'edgelist ok', 'locked MSE'

        # clean-up edges sub-folder
        if not(locked):
            try:
                ## cleaning up if there is an existing edge dir
                matname = os.path.join(self.pe.edgematpath, exp + '_' + name_database)
                import shutil
                shutil.rmtree(matname)
            except Exception:
                pass

        # 3- Doing the independence check for this set
        if not(locked) and self.pe.do_indepcheck:
            if not(os.path.isdir(self.pe.figpath)): os.mkdir(self.pe.figpath)
            txtname = os.path.join(self.pe.figpath, exp + '_dependence_' + name_database + note + '.txt')
            if not(os.path.isfile(txtname)) and not(os.path.isfile(txtname + '_lock')):
                open(txtname + '_lock', 'w').close() # touching
                self.log.info(' >> Doing check_independence on %s ', txtname)
                out = self.check_independence(self.cohistedges(edgeslist, symmetry=False, display=None), name_database, exp)
                f = open(txtname, 'w')
                f.write(out)
                f.close()
                #print(out)
                try:
                    os.remove(txtname + '_lock')
                except Exception as e:
                    self.log.error('Failed to remove lock file %s_lock, error : %s ', txtname, e)

        # 4- Doing the edge figures to check the edge extraction process
        if not(locked) and self.pe.do_edgedir:
            edgedir = os.path.join(self.pe.edgefigpath, exp + '_' + name_database)
            if not(os.path.isdir(self.pe.figpath)): os.mkdir(self.pe.figpath)
            if not(os.path.isdir(self.pe.edgefigpath)): os.mkdir(self.pe.edgefigpath)
            if not(os.path.isdir(edgedir)): os.mkdir(edgedir)

            N_image = edgeslist.shape[2]
            for index in np.random.permutation(np.arange(len(imagelist))):
                filename, croparea = imagelist[index]
                ext = 'png'

                figname = os.path.join(edgedir, filename.replace('.png', '') + str(croparea) + '.' + ext)
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    try:
                        open(figname + '_lock', 'w').close()
                        self.log.info('> redoing figure %s ', figname)
                        image, filename_, croparea_ = self.patch(name_database=name_database, filename=filename, croparea=croparea, do_whitening=self.pe.do_whitening)
                        if noise>0.: image += noise*image[:].std()*self.texture(filename=filename, croparea=croparea)
                        # if self.pe.do_whitening: image = self.whitening(image)
                        fig, ax = self.show_edges(edgeslist[:, :, index], image=image*1.)
                        # print(figname)
                        self.savefig(fig, figname)
                        try:
                            os.remove(figname + '_lock')
                        except Exception as e:
                            self.log.info('Failed to remove lock file %s_lock , error : %s ', figname , e)
                    except Exception as e:
                        self.log.info('Failed to make edge image  %s, error : %s ', figname , traceback.print_tb(sys.exc_info()[2]))

                figname = os.path.join(edgedir, filename.replace('.png', '') + str(croparea) + '_reconstruct' + '.' + ext)
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    try:
                        open(figname + '_lock', 'w').close()
                        self.log.info('> reconstructing figure %s ', figname)
                        image_ = self.reconstruct(edgeslist[:, :, index], do_mask=False)
#                         if self.pe.do_whitening: image_ = self.dewhitening(image_)
                        fig, ax = self.show_edges(edgeslist[:, :, index], image=image_*1.)
                        # print(figname)
                        self.savefig(fig, figname)
                        try:
                            os.remove(figname + '_lock')
                        except Exception as e:
                            self.log.error('Failed to remove lock file %s_lock, error : %s ', figname, traceback.print_tb(sys.exc_info()[2]))
                    except Exception as e:
                        self.log.error('Failed to make reconstruct image  %s , error : %s  ', figname, traceback.print_tb(sys.exc_info()[2]))

        if not(locked):
            # 6- Plotting the histogram and al
            try:
                ext = 'pdf'
                figname = os.path.join(self.pe.figpath, exp + '_proba-theta_' + name_database + note + '.' + ext)
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    open(figname + '_lock', 'w').close()
                    fig, ax = self.histedges_theta(edgeslist, display=True)
                    self.savefig(fig, figname, formats=[ext])
                    plt.close('all')
                    os.remove(figname + '_lock')
                #
                # figname = os.path.join(self.pe.figpath, exp + '_proba-edgefield_colin_' + name_database + note + '.' + ext)
                # if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                #     open(figname + '_lock', 'w').close()
                #     fig, ax = self.cohistedges(edgeslist, symmetry=False, display='colin_geisler')
                #     plt.savefig(figname)
                #     plt.close('all')
                #     os.remove(figname + '_lock')
                #
                # figname = os.path.join(self.pe.figpath, exp + '_proba-edgefield_cocir_' + name_database + note + '.' + ext)
                # if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                #     open(figname + '_lock', 'w').close()
                #     fig, ax = self.cohistedges(edgeslist, symmetry=False, display='cocir_geisler')
                #     plt.savefig(figname)
                #     plt.close('all')
                #     os.remove(figname + '_lock')

                figname = os.path.join(self.pe.figpath, exp + '_proba-edgefield_chevrons_' + name_database + note + '.' + ext)
                if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                    open(figname + '_lock', 'w').close()
                    fig, ax = self.cohistedges(edgeslist, display='chevrons')
                    self.savefig(fig, figname, formats=[ext])
                    plt.close('all')
                    os.remove(figname + '_lock')

                if 'targets' in name_database or 'laboratory' in name_database:
                    figname = os.path.join(self.pe.figpath, exp + '_proba-edgefield_chevrons_priordistractors_' + name_database + '_' + note + '.' + ext)
                    if not(os.path.isfile(figname)) and not(os.path.isfile(figname + '_lock')):
                        open(figname + '_lock', 'w').close()
                        imagelist_prior = self.get_imagelist(exp, name_database=name_database.replace('targets', 'distractors'))
                        edgeslist_prior = self.full_run(exp, name_database.replace('targets', 'distractors'), imagelist_prior, noise=noise)
                        v_hist_prior = self.cohistedges(edgeslist_prior, display=None)
                        fig, ax = self.cohistedges(edgeslist, display='chevrons', prior=v_hist_prior)
                        self.savefig(fig, figname, formats=[ext])
                        plt.close('all')
                        os.remove(figname + '_lock')
            except Exception as e:
                self.log.error('Failed to create figures, error : %s ', e)

        if not(locked):
            return imagelist, edgeslist, MSE
        else:
            return 'locked', 'locked edgeslist', ' locked MSE '

    # # some helper funtion to compare the databases
    # def KL(self, v_hist, v_hist_obs):
    #     """
    #     Computes the kullback-Leibler divergence  between 2 histograms
    #
    #     """
    #
    #     if v_hist.sum()==0 or v_hist_obs.sum()==0: self.log.error('>X>X>X KL function:  problem with null histograms! <X<X<X<')
    #     elif True:
    #         v_hist /= v_hist.sum()
    #         v_hist_obs /= v_hist_obs.sum()
    #         # taking advantage of log(True) = 0 and canceling out null bins in v_hist_obs
    #         kl = np.sum(v_hist.ravel()*(np.log(v_hist.ravel()+(v_hist == 0).ravel())
    #                                     - np.log(v_hist_obs.ravel()+(v_hist_obs == 0).ravel())))
    #         if kl == np.nan: print ( v_hist.sum(), v_hist_obs.sum() )
    #         return kl
    #     else:
    #         from scipy.stats import entropy
    #         return entropy(v_hist_obs, v_hist, base=2)

    def check_independence(self, v_hist, name_database, exp, labels=['d', 'phi', 'theta', 'scale']):
        v_hist /= v_hist.sum()
        fullset = [0, 1, 2, 3]
#    from scipy.stats import entropy
#    print KL(v_hist, v_hist),  entropy(v_hist.ravel(), v_hist.ravel())
        flat = np.ones_like(v_hist)
        flat /= flat.sum()
        out = 'Checking dependence in ' + name_database + '_' + exp + '\n'
        out += '-'*60 + '\n'
        out += 'Entropy: ' + str(KL(v_hist, flat.ravel())) + '\n'
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
        if ax==None: ax = fig.add_subplot(111, facecolor='w')
        # axes.edgecolor      : black   # axes edge color
        if (threshold==None) and (ref==None):
            if revert:
                inset = fig.add_subplot(111, facecolor='w')
                # this is another inset axes over the main axes
                ax = fig.add_axes([0.48, 0.55, .4, .4], facecolor='w')
            else:
                ax = fig.add_subplot(111, facecolor='w')
                # this is another inset axes over the main axes
                inset = fig.add_axes([0.48, 0.55, .4, .4], facecolor='w')
            #CCycle = np.vstack((np.linspace(0, 1, len(experiments)), np.zeros(len(experiments)), np.zeros(len(experiments)))).T
            grad = np.linspace(0., 1., 2*len(experiments))
            grad[1::2] = grad[::2]
            CCycle = np.array(color)[None, :] * grad[:, None]
            ax.set_color_cycle(CCycle)
            inset.set_color_cycle(CCycle)
            l0_max, eev = 0., -len(experiments)/2
            for mp, experiment, name_database, label in zip(mps, experiments, databases, labels):
                try:
                    imagelist, edgeslist, MSE = mp.process(exp=experiment, name_database=name_database)
                    # print(MSE.shape, MSE[:, 0])
                    N = MSE.shape[1] #number of edges
                    l0_max = max(l0_max, N*np.log2(mp.oc)/mp.pe.N_X/mp.pe.N_Y)
                    if not(scale):
                        l0_axis = np.arange(N)
                    else:
                        l0_axis = np.linspace(0, N*np.log2(mp.oc)/mp.pe.N_X/mp.pe.N_Y, N)
                    errorevery_zoom = 1.4**(1.*eev/len(experiments))
                    try:
                        MSE /= MSE[:, 0][:, None]
                        errorevery = np.max((int(MSE.shape[1]/8*errorevery_zoom), 1))
                        ax.errorbar(l0_axis, MSE.mean(axis=0),
                                    yerr=MSE.std(axis=0), label=label, errorevery=errorevery)
                        ax.plot(l0_axis[::errorevery], MSE.mean(axis=0)[::errorevery],
                                    linestyle='None', marker='o', ms=3)
                    except Exception as e:
                        print('Failed to plot MSE in experiment %s with error : %s ' % (experiment, e) )
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
                #ax.set_yscale("log")#, nonposx = 'clip')
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
                #ax.spines['left'].set_smart_bounds(True)
                a.spines['bottom'].set_smart_bounds(True)
                a.xaxis.set_ticks_position('bottom')
                a.yaxis.set_ticks_position('left')

                a.grid(b=False, which="both")

            if not(scale):#False and a==ax:
                ax.set_xlabel(r'$\ell_0$-norm')
            else:
                ax.set_xlabel(r'relative $\ell_0$ pseudo-norm (bits / pixel)')#relative $\ell_0$-norm')
            ax.set_ylim(-.02, 1.02)
            ax.set_ylabel(r'Squared error')
            inset.set_ylabel(r'Coefficient')
            if revert:
                ax.legend(loc='best', frameon=False, prop={'size': 6})#, bbox_to_anchor = (0.5, 0.5))
            else:
                inset.legend(loc='best', frameon=False, bbox_to_anchor = (0.4, 0.4), prop={'size': 6})
            plt.locator_params(tight=False, nbins=4)
            plt.tight_layout()
            return fig, ax, inset
        elif (threshold==None):
            if ax==None: ax = fig.add_axes([0.15, 0.25, .75, .75], facecolor='w')
            ind, l0, l0_std = 0, [], []
            from lmfit.models import ExpressionModel
            mod = ExpressionModel('1 - (1- eps_inf) * ( 1 - rho**x)')
            for mp, experiment, name_database, label in zip(mps, experiments, databases, labels):
                try:
                    imagelist, edgeslist, MSE = mp.process(exp=experiment, name_database=name_database)
                    MSE /= MSE[:, 0][:, None]
                    N = MSE.shape[1] #number of edges
                    if MSE.min()>threshold: print('the threshold is never reached for', experiment, name_database)
                    try:
                        l0_results = np.zeros(N)
                        for i_image in range(MSE.shape[0]):
                            mod.def_vals = {'eps_inf':.1, 'rho':.99}
                            out  = mod.fit(MSE[i_image, :], x=np.arange(N))
                            eps_inf = out.params.get('eps_inf').value
                            rho =  out.params.get('rho').value
                            #print rho, eps_inf, np.log((threshold-eps_inf)/(1-eps_inf))/np.log(rho)

                            l0_results[i_image] = np.log((threshold-eps_inf)/(1-eps_inf))/np.log(rho)
                    except Exception:
                        l0_results = np.argmax(MSE<threshold, axis=1)*1.
                    if (scale):
                        l0_results *= np.log2(mp.oc)/mp.pe.N_X/mp.pe.N_Y
                    l0.append(l0_results.mean())
                    l0_std.append(l0_results.std())
                    ind += 1
                except Exception as e:
                    print('Failed to plot (no threshold) experiments %s with error : %s ' % (experiments, e) )


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
                imagelist, edgeslist, MSE = mp.process(exp=experiment, name_database=name_database)
                if isinstance(MSE, str):
                    print ('not finished in ', experiment, name_database)
                    return None
                MSE /= MSE[:, 0][:, None]
                N = MSE.shape[1] #number of edges
                l0 = np.log2(mp.oc)/mp.pe.N_X/mp.pe.N_Y
                absSE.append(MSE.mean())
                absSE_std.append(MSE.std(axis=0).mean())
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

            ax.set_xticks(np.arange(ind))
            ax.set_xticklabels(labels)

            plt.tight_layout()
#         fig.set_tight_layout(True)

            return fig, ax, ax

        else: # fourth type: we have a reference and a threshold
            try:
                relL0, relL0_std = [], []
                # computes for the reference
                imagelist_ref, edgeslist_ref, MSE_ref = mps[ref].process(exp=experiments[ref], name_database=databases[ref])
                MSE_ref /= MSE_ref[:, 0][:, None] # normalize MSE
                L0_ref =  np.argmax(MSE_ref<threshold, axis=1)*1. +1
                if scale: L0_ref *= np.log2(mps[ref].oc)/mps[ref].pe.N_X/mps[ref].pe.N_Y
#             print("ref-thr - L0_ref=", L0_ref)

                for mp, experiment, name_database, label in zip(mps, experiments, databases, labels):
                    imagelist, edgeslist, MSE = mp.process(exp=experiment, name_database=name_database)
                    MSE /= MSE[:, 0][:, None] # normalize MSE
                    N = MSE.shape[1] #number of edges
                    L0 =  np.argmax(MSE<threshold, axis=1)*1.
                    if MSE.min()>threshold: print('the threshold is never reached for', experiment, name_database)
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
                # ax.set_xlim([-width/2, ind+.0*width])

                ax.set_ylabel(r'relative coding cost wrt default')# (bits / pixel)')#relative $\ell_0$-norm')

                ax.set_xticks(np.arange(ind))
                ax.set_xticklabels(labels)

                ax.grid(b=False, which="both")
#         plt.tight_layout()
#         fig.set_tight_layout(True)

                return fig, ax, ax

            except Exception as e:
                print('Failed to analyze experiment %s with error : %s ' % (experiment, e) )

def KL(v_hist_ref, v_hist_obs, p0=1.e-3):
    """
    Computes the kullback-Leibler divergence between 2 histograms
    the histogram is represented in the last dimension
    """
    if v_hist_ref.sum()==0 or v_hist_obs.sum()==0:
        print('ddooooh')
        return 10000

    v_hist_ref = v_hist_ref + p0 / v_hist_ref.shape[-1]
    v_hist_obs = v_hist_obs + p0 / v_hist_obs.shape[-1]

    if v_hist_ref.ndim == 2:
        v_hist_ref /= v_hist_ref.sum(axis=-1)[:, None]
        v_hist_obs /= v_hist_obs.sum(axis=-1)[:, None]
        v_hist_ref = v_hist_ref[:, None, :]
        v_hist_obs = v_hist_obs[None, :, :]
    else: # one vector
        v_hist_ref /= v_hist_ref.sum()
        v_hist_obs /= v_hist_obs.sum()

    # taking advantage of log(True) = 0 and canceling out null bins in v_hist_obs
    # v_hist_ref_ = v_hist_ref + (v_hist_ref == 0.)
    # v_hist_obs_ = v_hist_obs + (v_hist_obs == 0.)

    return np.sum(v_hist_ref * (np.log2(v_hist_ref/v_hist_obs)), axis=-1)

def TV(v_hist_ref, v_hist_obs):
    """
    Computes the TV distance between 2 histograms

    the histogram is represented in the last dimension

    https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures

    """
    if v_hist_ref.ndim == 2:
        v_hist_ref /= v_hist_ref.sum(axis=-1)[:, None]
        v_hist_obs /= v_hist_obs.sum(axis=-1)[:, None]
        v_hist_ref = v_hist_ref[:, None, :]
        v_hist_obs = v_hist_obs[None, :, :]
    else: # one vector
        v_hist_ref /= v_hist_ref.sum()
        v_hist_obs /= v_hist_obs.sum()

    return np.max(np.abs(v_hist_ref-v_hist_obs), axis=-1)


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
