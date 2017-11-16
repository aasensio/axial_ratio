import numpy as np
import matplotlib.pyplot as pl
import pymc3 as pm
import theano.tensor as tt
from ipdb import set_trace as stop

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

    def mock(self, muB, sigmaB, muC, sigmaC, noise, n_galaxies, phi_limit=np.pi/2.0):
        """
        Generate a mock dataset
        """

# Define number of galaxies in the mock dataset and the uncertainty in the observation of q
        self.n_galaxies = n_galaxies
        self.sigmaq = noise

# Galaxy orientation is isotropic in space. mu=cos(theta)
        self.mu = np.random.rand(self.n_galaxies)
        self.phi = phi_limit * np.random.rand(self.n_galaxies)

# Generate truncated normal distributions using PYMC3
        tmpB = pm.Bound(pm.Normal, lower=0.0, upper=1.0)
        tmpC = pm.Bound(pm.Normal, lower=0.0, upper=1.0)

# Sample from these distributions
        B = tmpB.dist(mu=muB, sd=sigmaB).random(size=self.n_galaxies)
        C = tmpC.dist(mu=muC, sd=sigmaC).random(size=self.n_galaxies)

        BC = np.vstack([B, C])
        BC = np.sort(BC, axis=0)
        self.C, self.B = BC[0,:], BC[1,:]

# Compute axial ratios
        q = self.axial_ratio(self.B, self.C, self.mu, self.phi)

# Generate fake observations by adding Gaussian noise
        self.qobs = q + self.sigmaq * np.random.randn(self.n_galaxies)

    def read_obs(self, file):
        tmp = np.loadtxt(file)
        self.sigmaq = 0.04
        self.n_galaxies = len(tmp)
        self.qobs = tmp

    def sample(self, name_chain):
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

        self.name_chain = name_chain

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

# Finally sample from the posterior and use a CSV backend for later plots
            db = pm.backends.Text(self.name_chain)
            self.trace = pm.sample(chains=4, trace=db)
            self.ppc = pm.sample_ppc(self.trace, samples=500, model=self.model, size=100)

if (__name__ == '__main__'):
    pl.close('all')
    out = galaxy()

    out.read_obs('data/all.dat')
    out.sample('data/all.samples')

    f, ax = pl.subplots(nrows=2, ncols=2, figsize=(8,8))
    ax[0,0].plot(out.trace['muCB'][:,0],out.trace['muCB'][:,1],'.',alpha=0.2)
    ax[0,0].set_xlabel('muC')
    ax[0,0].set_ylabel('muB')
    ax[0,1].plot(out.trace['muCB'][:,0],out.trace['sdCB'][:,1],'.',alpha=0.2)
    ax[0,1].set_xlabel('muC')
    ax[0,1].set_ylabel('sdB')
    ax[1,1].plot(out.trace['sdCB'][:,0],out.trace['sdCB'][:,1],'.',alpha=0.2)
    ax[1,1].set_xlabel('sdC')
    ax[1,1].set_ylabel('sdB')
    ax[1,0].plot(out.trace['sdCB'][:,0],out.trace['muCB'][:,1],'.',alpha=0.2)
    ax[1,0].set_xlabel('sdC')
    ax[1,0].set_ylabel('muB')
    pl.tight_layout()
    pl.show()

    f, ax = pl.subplots()
    ax.hist(out.ppc['qobs'].flatten(), histtype='step', density=True)
    ax.hist(out.qobs, histtype='step', density=True)
    pl.show()
