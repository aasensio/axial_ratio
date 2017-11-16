import numpy as np
import matplotlib.pyplot as pl
import pymc3 as pm
import theano.tensor as tt
import corner

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

    def sample(self):
        """
        Sample from the hierarchical model
        p(mu) ~ U(0,1)
        p(phi) ~ U(0,pi)
        p(muB) ~ U(0.0, 1.0)
        p(muC) ~ U(0.0, 1.0)
        p(sigmaB) ~ HN(sd=1)
        p(sigmaC) ~ HN(sd=1)
        p(B|muB,sigmaB) ~ BoundedNormal(mu=muB, sd=sigmaB, lower=0, upper=1)
        p(C|muC,sigmaC) ~ BoundedNormal(mu=muC, sd=sigmaC, lower=0, upper=1)
        qobs ~ N(mu=q, sd=noise)
        q = f(B,C,mu,phi)

        """

        A = 1.0

# Define the probabilistic model
        self.model = pm.Model()
        with self.model:

# Priors for orientation
            mu = pm.Uniform('mu', lower=0, upper=1.0, testval=0.5, shape=self.n_galaxies)
            phi = pm.Uniform('phi', lower=0, upper=np.pi / 2.0, testval=0.1, shape=self.n_galaxies)

# Priors for means and standard deviations. Perhaps one should play a little with the
# priors for sdB and sdC because they are usually not very well constrained by data
# One should also consider using a non-centered model: http://twiecki.github.io/blog/2017/02/08/bayesian-hierchical-non-centered/
            muCB_ = pm.Uniform('muCB_', lower=0.0, upper=1.0, testval=[0.3, 0.8], shape=2)
            muCB = pm.Deterministic('muCB', tt.sort(muCB_))

            sdCB = pm.HalfNormal('sdCB', sd=0.05, shape=2)

            bounded_normal = pm.Bound(pm.Normal, lower=0.0, upper=1.0)
            CB_ = bounded_normal('CB_', mu=muCB, sd=sdCB, testval=np.array([0.3,0.8]), shape=(self.n_galaxies,2))                        
            CB = pm.Deterministic('CB', tt.sort(CB_, axis=1))
          
# Now that we have all ingredients, compute q
            sin_theta = tt.sqrt(1.0 - mu**2)
            f = ( A*CB[:,0]*sin_theta*tt.cos(phi) )**2 + ( CB[:,1]*CB[:,0]*sin_theta*tt.sin(phi) )**2 + ( A*CB[:,1]*mu )**2 
            g = A*A * (tt.cos(phi)**2 + mu**2 * tt.sin(phi)**2) + \
                CB[:,1]*CB[:,1] * (tt.sin(phi)**2 + mu**2 * tt.cos(phi)**2) + CB[:,0]*CB[:,0] * sin_theta**2    

            h = tt.sqrt(  (g - 2 * tt.sqrt(f)) / (g + 2 * tt.sqrt(f))  )
            q = (1 - h) / (1 + h)

# And define the normal likelihood
            qobs = pm.Normal('qobs', mu=q, sd=self.sigmaq, observed=self.qobs, shape=self.n_galaxies)

    def doplot(self, name):
        """
        Do some plots
        """
        with self.model:
            trace = pm.backends.text.load(name)

        var = np.vstack([trace['muCB'][:,0], trace['muCB'][:,1], trace['sdCB'][:,0], trace['sdCB'][:,1]]).T

        corner.corner(var, labels=['$\mu_C$', '$\mu_B$', '$\sigma_C$','$\sigma_B$'], show_titles=True)
        
        pl.show()

        pl.savefig('{0}.png'.format(name))


if (__name__ == '__main__'):
    pl.close('all')
    out = galaxy()

    out.read_obs('data/big.dat')
    out.sample()
    out.doplot('data/big.samples')

    # out.read_obs('data/small.dat')
    # out.sample()
    # out.doplot('data/small.samples')

    # out.read_obs('data/all.dat')
    # out.sample()
    # out.doplot('data/all.samples')