# -*- coding: utf8 -*-
"""
SparseEdges

See http://invibe.net/LaurentPerrinet/Publications/Perrinet11sfn

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
import numpy as np
import scipy.ndimage as nd
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

# import LogGabor.LogGabor as LogGabor
# import LogGabor.Image as Image
# 

def init_pylab():
    ############################  FIGURES   ########################################
    from NeuroTools import check_dependency
    HAVE_MATPLOTLIB = check_dependency('matplotlib')
    if HAVE_MATPLOTLIB:
        import matplotlib
        matplotlib.use("Agg") # agg-backend, so we can create figures without x-server (no PDF, just PNG etc.)
    HAVE_PYLAB = check_dependency('pylab')
    if HAVE_PYLAB:
        import pylab
        # parameters for plots
        fig_width_pt = 500.  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inches
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fontsize = 8
        # pe.edge_scale_chevrons, line_width = 64., .75
        params = {'backend': 'Agg',
                 'origin': 'upper',
                  'font.family': 'serif',
                  'font.serif': 'Times',
                  'font.sans-serif': 'Arial',
                  'text.usetex': True,
        #          'mathtext.fontset': 'stix', #http://matplotlib.sourceforge.net/users/mathtext.html
                  'interpolation':'nearest',
                  'axes.labelsize': fontsize,
                  'text.fontsize': fontsize,
                  'legend.fontsize': fontsize,
                  'figure.subplot.bottom': 0.17,
                  'figure.subplot.left': 0.15,
                  'ytick.labelsize': fontsize,
                  'xtick.labelsize': fontsize,
                  'savefig.dpi': 100,
                }
        pylab.rcParams.update(params)

def adjust_spines(ax,spines):
    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            spine.set_position(('outward',1)) # outward by 10 points
#            spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
class MatchingPursuit:
    """
    defines a MatchingPursuit algorithm

    """
    def __init__(self, lg):
        """
        initializes the LogGabor structure

        """
        self.pe = lg.pe
        self.MP_alpha = self.pe.MP_alpha

        self.lg = lg
        self.im = lg.im
        self.n_x = lg.n_x
        self.n_y = lg.n_y

        self.base_levels = self.pe.base_levels
        self.n_levels = int(np.log(np.max((self.n_x, self.n_y)))/np.log(self.base_levels)) #  self.pe.n_levels
        self.MP_alpha = self.pe.MP_alpha

        self.sf_0 = lg.n_x / np.logspace(1, self.n_levels, self.n_levels, base=self.base_levels)

        self.n_theta = self.pe.n_theta
        self.theta_ = np.linspace(0., np.pi, self.n_theta, endpoint=False)
        self.B_theta = self.pe.B_theta
        self.B_sf = self.pe.B_sf
        self.N = self.pe.N
        self.do_whitening = self.pe.do_whitening
        self.do_mask = self.pe.do_mask
        if self.do_mask:
            X, Y = np.mgrid[-1:1:1j*self.n_x, -1:1:1j*self.n_y]
            self.mask = (X**2 + Y**2) < 1.

    def run(self, image, verbose=False):
        edges = np.zeros((5, self.N), dtype=np.complex)
        image_ = image.copy()
        if self.do_whitening: image_ = self.im.whitening(image_)
        C = self.init(image_)
        for i_edge in range(self.N):
            # MATCHING
            ind_edge_star = self.argmax(C)
            # recording
            if verbose: print 'Max activity  : ', np.absolute(C[ind_edge_star]), ' phase= ', np.angle(C[ind_edge_star], deg=True), ' deg,  @ ', ind_edge_star
            edges[:, i_edge] = np.array([ind_edge_star[0]*1., ind_edge_star[1]*1., self.theta_[ind_edge_star[2]], self.sf_0[ind_edge_star[3]], self.MP_alpha * C[ind_edge_star]])
            # PURSUIT
            C = self.backprop(C, ind_edge_star)
#            if verbose: print 'Residual activity : ',  C[ind_edge_star]
        return edges, C
#
    def init(self, image):
        C = np.empty((self.n_x, self.n_y, self.n_theta, self.n_levels), dtype=np.complex)
        for i_sf_0, sf_0 in enumerate(self.sf_0):
            for i_theta, theta in enumerate(self.theta_):
                FT_lg = self.lg.loggabor(0, 0, sf_0=sf_0, B_sf=self.B_sf,
                                    theta=theta, B_theta=self.B_theta)
                C[:, :, i_theta, i_sf_0] = self.im.FTfilter(image, FT_lg, full=True)
                if self.do_mask: C[:, :, i_theta, i_sf_0] *= self.mask
        return C

    def reconstruct(self, edges):
#        Fimage = np.zeros((self.n_x, self.n_y), dtype=np.complex)
        image = np.zeros((self.n_x, self.n_y))
#        print edges.shape, edges[:, 0]
        for i_edge in range(edges.shape[1]):#self.N):
            # TODO : check that it is correct when we remove alpha when making new MP
            image += self.im.invert(edges[4, i_edge] * self.lg.loggabor(
                                                                        edges[0, i_edge].real, edges[1, i_edge].real,
                                                                        theta=edges[2, i_edge].real, B_theta=self.B_theta,
                                                                        sf_0=edges[3, i_edge].real, B_sf=self.B_sf,
                                                                        ),
                                    full=False)
        return image

    def argmax(self, C):
        """
        Returns the ArgMax from C by returning the
        (x_pos, y_pos, theta, scale)  tuple

        >>> C = np.random.randn(10, 10, 5, 4)
        >>> C[x_pos][y_pos][level][level] = C.max()

        """
        ind = np.absolute(C).argmax()
        return np.unravel_index(ind, C.shape)

    def backprop(self, C, edge_star):
        """
        Removes edge_star from the activity

        """
        C_star = self.MP_alpha * C[edge_star]
        FT_lg_star = self.lg.loggabor(edge_star[0]*1., edge_star[1]*1., sf_0=self.sf_0[edge_star[3]],
                         B_sf=self.B_sf,#_ratio*self.sf_0[edge_star[3]],
                    theta= self.theta_[edge_star[2]], B_theta=self.B_theta)
        lg_star = self.im.invert(C_star*FT_lg_star, full=False)

        for i_sf_0, sf_0 in enumerate(self.sf_0):
            for i_theta, theta in enumerate(self.theta_):
                FT_lg = self.lg.loggabor(0, 0, sf_0=sf_0, B_sf=self.B_sf, theta=theta, B_theta=self.B_theta)
                C[:, :, i_theta, i_sf_0] -= self.im.FTfilter(lg_star, FT_lg, full=True)
                if self.do_mask: C[:, :, i_theta, i_sf_0] *= self.mask
        return C

    def adapt(self, edges):
        # TODO : implement a COMP adaptation of the thetas and scales tesselation of Fourier space
        pass

    def show_edges(self, edges, fig=None, a=None, image=None, norm=True,
                   color='auto', v_min=-1., v_max=1., show_phase=False, gamma=1., pedestal=.2, mappable=False):
        """
        Shows the quiver plot of a set of edges, optionally associated to an image.

        """
        import pylab
        import matplotlib.cm as cm
        if fig==None:
            fig = pylab.figure(figsize=(self.pe.figsize_edges, self.pe.figsize_edges))
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

        opts= {'extent': (0, self.n_x, self.n_y, 0),
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

            X, Y, Theta, Sf_0 = edges[1, :].real+.5, edges[0, :].real+.5, np.pi -  edges[2, :].real, edges[3, :].real
            weights = edges[4, :]

            #show_phase, pedestal = False, .2 # color edges according to phase or hue? pedestal value for alpha when weights= 0

    #        print X, Y, Theta, Sf_0, weights, scale_
    #        print 'Min theta ', Theta.min(), ' Max theta ', Theta.max()
#            weights = np.absolute(weights)/(np.abs(weights)).max()
            weights = weights/(np.abs(weights)).max()

            for x, y, theta, sf_0, weight in zip(X, Y, Theta, Sf_0, weights):
                u_, v_ = np.cos(theta)*scale/sf_0*self.n_x, np.sin(theta)*scale/sf_0*self.n_y
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
            n_ = np.sqrt(self.n_x**2+self.n_y**2)
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
                circ = patches.Circle((x,y), self.pe.scale_circle*scale/sf_0*n_, facecolor=fc, edgecolor='none')#, alpha=0.5*weight)
                # (0.5, 0.5), 0.25, transform=ax.transAxes, facecolor='yellow', alpha=0.5)
                a.add_patch(circ)

            line_segments = LineCollection(segments, linewidths=linewidths, colors=colors, linestyles='solid')
            a.add_collection(line_segments)

        if not(color=='auto'):# chevrons maps etc...
            pylab.setp(a, xticks=[])
            pylab.setp(a, yticks=[])

        a.axis([0, self.n_x, self.n_y, 0])
        pylab.draw()
        if mappable:
            return fig, a, line_segments
        else:
            return fig, a


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
#     from pylab import imread
#     # whitening
#     image = imread('database/gris512.png')[:,:,0]
#     lg = LogGabor(image.shape)


