import numpy as np
import matplotlib.pyplot as pl
import pystan
import pickle
from hashlib import md5

from ipdb import set_trace as stop

def stan_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm

code_simple = """

data {
    int<lower=0> N;
    vector[N] qobs;
    vector[N] sigmaq;
}

parameters {    
    real<lower=0> sdCB[2];
    vector<lower=0,upper=1>[N] mu;
    vector<lower=0,upper=pi()/2>[N] phi;
    simplex[3] CB_[N];
    simplex[3] muCB_;
}

transformed parameters {    
    positive_ordered[2] muCB;
    positive_ordered[2] CB[N];
    muCB = head(cumulative_sum(muCB_), 2);
    for (i in 1:N) {
        CB[i] = head(cumulative_sum(CB_[i]), 2);
    }
}

model {
    real q[N];
    real A=1.0;
    real sin_theta;
    real f;
    real g;
    real h;

    sdCB ~ normal(0.0, 0.05);

    for (i in 1:N) {
        CB[i] ~ normal(muCB, sdCB);
    }

    for (i in 1:N) {
        sin_theta = sqrt(1.0 - square(mu[i]));
        f = square( A*CB[i,1]*sin_theta*cos(phi[i]) ) + square( CB[i,2]*CB[i,1]*sin_theta*sin(phi[i]) ) 
            + square( A*CB[i,2]*mu[i] );
        g = A*A * (square(cos(phi[i])) + square(mu[i]) * square(sin(phi[i])) ) + 
            CB[i,2]*CB[i,2] * (square(sin(phi[i])) + square(mu[i]) * square(cos(phi[i]))) + 
            CB[i,1]*CB[i,1] * square(sin_theta);

        h = sqrt(  (g - 2 * sqrt(f)) / (g + 2 * sqrt(f))  );
        q[i] = (1 - h) / (1 + h);

        qobs[i] ~ normal(q[i], sigmaq[i]) T[0,1];
    }    
}

"""

code_mixture = """

data {
    int<lower=0> N;
    vector[N] qobs;
    vector[N] sigmaq;
}

parameters {    
    real<lower=0> sdCB[2];
    vector<lower=0,upper=1>[N] mu;
    vector<lower=0,upper=pi()/2>[N] phi;
    simplex[3] CB_[N];
    simplex[3] muCB_;
}

transformed parameters {    
    positive_ordered[2] muCB;
    positive_ordered[2] CB[N];
    muCB = head(cumulative_sum(muCB_), 2);
    for (i in 1:N) {
        CB[i] = head(cumulative_sum(CB_[i]), 2);
    }
}

model {
    real q[N];
    real A=1.0;
    real sin_theta;
    real f;
    real g;
    real h;

    sdCB ~ normal(0.0, 0.05);

    for (i in 1:N) {
        CB[i] ~ normal(muCB, sdCB);
    }

    for (i in 1:N) {
        sin_theta = sqrt(1.0 - square(mu[i]));
        f = square( A*CB[i,1]*sin_theta*cos(phi[i]) ) + square( CB[i,2]*CB[i,1]*sin_theta*sin(phi[i]) ) 
            + square( A*CB[i,2]*mu[i] );
        g = A*A * (square(cos(phi[i])) + square(mu[i]) * square(sin(phi[i])) ) + 
            CB[i,2]*CB[i,2] * (square(sin(phi[i])) + square(mu[i]) * square(cos(phi[i]))) + 
            CB[i,1]*CB[i,1] * square(sin_theta);

        h = sqrt(  (g - 2 * sqrt(f)) / (g + 2 * sqrt(f))  );
        q[i] = (1 - h) / (1 + h);
    }

    qobs ~ normal(q, sigmaq) T[0,1];
}

"""

def read_obs(filename):
    tmp = np.loadtxt(filename)
    if (tmp.ndim == 1):            
        n_galaxies = len(tmp)
        sigmaq = np.ones(n_galaxies) * 0.04
        qobs = tmp
    if (tmp.ndim == 2):
        n_galaxies, _ = tmp.shape
        qobs = tmp[:,0]
        sigmaq = tmp[:,1]

    return qobs, sigmaq
    
if (__name__ == '__main__'):

    qobs, sigmaq = read_obs('data/small_plus_uncer.dat')
    N = len(qobs)

    
    data = {'N': N, 'qobs': qobs, 'sigmaq': sigmaq}

    model = stan_cache(model_code=code_simple)

    fit = model.sampling(data=data)

    samples = fit.extract(permuted=True)

    pickle.dump( samples, open( "data/small_plus_uncer.samples.pickle", "wb" ) )
  

# model = stan_cache(model_code=code)
# init = {'a1': 0.8*np.ones(nx*ny), 'b1': np.zeros(nx*ny), 'c1': 15.0*np.ones(nx*ny), 'a2': 0.3*np.ones(nx*ny), 'b2': np.zeros(nx*ny), 'c2': 10.0*np.ones(nx*ny), 
#     'cont': np.zeros(nx*ny), 'l_gp': 2.0*np.ones(2), 'eta': 0.2}

# vb = model.vb(data=data, eta=0.1, init=init, adapt_engaged=False, sample_file='./variational.sample', diagnostic_file='./variational.diag', tol_rel_obj=0.001, 
#     iter=100000, pars=['a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'cont'])
# out = pd.read_csv(vb['args']['sample_file'].decode("utf-8"), header=7).as_matrix()[:,1:]

# n_samples, _ = out.shape
# samples = out[:,0:7*nx*ny].reshape((n_samples,7,nx,ny))

# f, ax = pl.subplots(ncols=2)
# ax[0].plot(out[:,7*nx*ny])
# ax[0].plot(out[:,7*nx*ny+1])
# ax[0].set_ylabel('l_gp')

# ax[1].plot(out[:,7*nx*ny+2])
# ax[1].set_ylabel('eta')


# f, ax = pl.subplots(nrows=2, ncols=7, figsize=(12,7))
# for i in range(7):
#     ax[0,i].imshow(np.mean(samples[-100:,i,:,:], axis=0))
#     ax[1,i].imshow(np.median(samples[-100:,i,:,:], axis=0))

# pl.show()

# f, ax = pl.subplots(nrows=4, ncols=4, figsize=(12,8))

# for i in range(4):
#     for j in range(4):
#         a1 = samples[-100:,0,i,j][:,None]
#         b1 = samples[-100:,1,i,j][:,None]
#         c1 = samples[-100:,2,i,j][:,None]
#         a2 = samples[-100:,3,i,j][:,None]
#         b2 = samples[-100:,4,i,j][:,None]
#         c2 = samples[-100:,5,i,j][:,None]
#         cont = samples[-100:,6,i,j][:,None]
#         ymodel = cont + a1*np.exp(-c1**2 * (velocity[None,:]-b1)**2) + a2*np.exp(-c2**2 * (velocity[None,:]-b2)**2)
#         ax[i,j].plot(cube[:,i,j])
#         ax[i,j].plot(ymodel.T, color='C1', alpha=0.1)
# pl.show()