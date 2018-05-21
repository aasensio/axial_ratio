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

    # Log likelihood of Gaussian mixture distribution
    def logp_gmix(mus, pi, sd):
        def logp_(value):
            logps = [tt.log(pi[i]) + logp_normal(mu, sd, value)
                 for i, mu in enumerate(mus)]

            return tt.sum(logsumexp(tt.stacklists(logps)[:, :], axis=0))

        return logp_

    def sample(self, name_chain, noncentered=False):
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
            muCB_ = pm.Uniform('muCB_', lower=0.0, upper=1.0, testval=[0.3, 0.8], shape=2)
            muCB = pm.Deterministic('muCB', tt.sort(muCB_))

            muCB2_ = pm.Uniform('muCB2_', lower=0.0, upper=1.0, testval=[0.3, 0.8], shape=2)
            muCB2 = pm.Deterministic('muCB2', tt.sort(muCB2_))

            sdCB = pm.HalfNormal('sdCB', sd=0.05, shape=2)
            sdCB2 = pm.HalfNormal('sdCB2', sd=0.05, shape=2)

            w = pm.Dirichlet('w', np.ones(2))

# Use a non-centered model (http://twiecki.github.io/blog/2017/02/08/bayesian-hierchical-non-centered/)
            if (noncentered):
                offset = pm.Normal('offset', mu=0, sd=1, shape=(self.n_galaxies,2))
                CB_ = pm.Deterministic('CB_', tt.clip(muCB + offset * sdCB, 0.0, 1.0))
                CB = pm.Deterministic('CB', tt.sort(CB_, axis=1))
            else: 
                
                bounded_normal = pm.Bound(pm.Normal, lower=0.0, upper=1.0)
                CB_ = bounded_normal.dist('CB_', mu=muCB, sd=sdCB, testval=np.array([0.3,0.8]), shape=(self.n_galaxies,2))            
                CB2_ = bounded_normal.dist('CB2_', mu=muCB2, sd=sdCB2, testval=np.array([0.3,0.8]), shape=(self.n_galaxies,2))
                comp_dists = [CB_, CB2_]

                mix = pm.Mixture('mix', w=w, comp_dists = comp_dists, testval=np.array([0.3,0.8]), shape=(self.n_galaxies, 2))               
                
                CB = pm.Deterministic('CB', tt.sort(mix, axis=1))
          
# Now that we have all ingredients, compute q
#             sin_theta = tt.sqrt(1.0 - mu**2)
#             f = ( A*CB[:,0]*sin_theta*tt.cos(phi) )**2 + ( CB[:,1]*CB[:,0]*sin_theta*tt.sin(phi) )**2 + ( A*CB[:,1]*mu )**2 
#             g = A*A * (tt.cos(phi)**2 + mu**2 * tt.sin(phi)**2) + \
#                 CB[:,1]*CB[:,1] * (tt.sin(phi)**2 + mu**2 * tt.cos(phi)**2) + CB[:,0]*CB[:,0] * sin_theta**2    

#             h = tt.sqrt(  (g - 2 * tt.sqrt(f)) / (g + 2 * tt.sqrt(f))  )
#             q = (1 - h) / (1 + h)

# # And define the normal likelihood
#             qobs = pm.Normal('qobs', mu=q, sd=self.sigmaq, observed=self.qobs, shape=self.n_galaxies)

# Finally sample from the posterior and use a CSV backend for later plots
            db = pm.backends.Text(self.name_chain)
            self.trace = pm.sample(chains=4, trace=db)
            self.ppc = pm.sample_ppc(self.trace, samples=500, model=self.model, size=100)

if (__name__ == '__main__'):
    pl.close('all')
    out = galaxy()

    out.read_obs('data/small_plus_uncer.dat')
    out.sample('data/small_plus_uncer.samples', noncentered=False)

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
