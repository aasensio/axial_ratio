import numpy as np
import matplotlib.pyplot as pl
import pymc3 as pm
import theano.tensor as tt
import corner
import seaborn as sns
import pickle


class galaxy(object):
    def __init__(self):
        """
        Class constructor.
        """
        pass

    def axial_ratio(self, B, C, cos_theta, phi):
        """
        Return the axial ratio for given combinations of B, C, cos(theta) and phi
        """
        A = 1
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        f = ( A*C*sin_theta*np.cos(phi) )**2 + ( B*C*sin_theta*np.sin(phi) )**2 + ( A*B*cos_theta )**2 
        g = A*A * (np.cos(phi)**2 + cos_theta**2 * np.sin(phi)**2) + \
            B*B * (np.sin(phi)**2 + cos_theta**2 * np.cos(phi)**2) + C*C * sin_theta**2    
        h = (  (g - 2 * f**0.5) / (g + 2 * f**0.5)  )**0.5
        q = (1 - h) / (1 + h)
        return q

    def read_obs(self, file):
        """
        Read the observations, that will be in a file with one or two columns
        If only one column is present, use 0.04 as an estimated uncertainty for the axial ratio
        """
        tmp = np.loadtxt(file)
        if (tmp.ndim == 1):            
            self.n_galaxies = len(tmp)
            self.sigmaq = np.ones(self.n_galaxies) * 0.04
            self.qobs = tmp
        if (tmp.ndim == 2):
            self.n_galaxies, _ = tmp.shape
            self.qobs = tmp[:,0]
            self.sigmaq = tmp[:,1]


    def doplot(self, name):
        """
        Do some plots
        """

        self.trace = pickle.load( open( name, "rb" ) )

        var = np.vstack([self.trace['muCB'][:,0], self.trace['muCB'][:,1], self.trace['sdCB'][:,0], self.trace['sdCB'][:,1]]).T

        corner.corner(var, labels=['$\mu_C$', '$\mu_B$', '$\sigma_C$','$\sigma_B$'], show_titles=True)
        
        pl.show()

        # pl.savefig('{0}.png'.format(name))

        # Just get the first N samples. We shuffle the
        # arrays and get the subsamples
        C = self.trace['CB'][:,:,0]
        np.random.shuffle(C)
        C_slice = C[0:200,:].flatten()
        B = self.trace['CB'][:,:,1]
        np.random.shuffle(B)
        B_slice = B[0:200,:].flatten()

        # First option
        pl.plot(B_slice, C_slice, '.', alpha=0.002)
        pl.show()

        # KDE joint plot
        sns.jointplot(C_slice, B_slice, kind='kde')
        pl.show()


if (__name__ == '__main__'):
    pl.close('all')
    out = galaxy()

    out.read_obs('data/big.dat')
    out.doplot('data/small_plus_uncer.samples.pickle')

    # out.read_obs('data/small.dat')
    # out.sample()
    # out.doplot('data/small.samples')

    # out.read_obs('data/all.dat')
    # out.sample()
    # out.doplot('data/all.samples')
