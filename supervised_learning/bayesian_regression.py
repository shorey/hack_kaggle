import numpy as np
from scipy.stats import chi2, multivariate_normal
from utils import mean_squared_error,train_test_split,polynomial_features


class BayesianRegression(object):
    """
    bayesian regression model. if poly_degree is specified the features
    will be transformed to with a polynomial basis function, which allows for
    ploynomial regression.
    assumes nomal prior and likelihood for the weights and scaled inverse
    chi-squared prior and likelihood for the variance of the weights.

    parameters:
    -------------
    n_draws: float
        the number of simulated draws from the posterior of the parameters.
    mu0: array
        the mean values of the prior Normal distribution of the parameters.
    omega0: array
        the precision matrix of the prior Normal distribution of the parameters

    nu0: float
        the degrees of freedom of the prior scaled inverse chi squared distribution

    sigma_sq0: float
        the scale parameter of the prior scaled inverse chi squared distribution

    poly_degree: int
        the polynomial degree that the features should be transformed to.allows
        for polynomial regression.
    cred_int:float
        the credible interval(ETI in this impl.). 95=>95% credible interval of the
        parameters.

         Reference:
        https://github.com/mattiasvillani/BayesLearnCourse/raw/master/Slides/BayesLearnL5.pdf
    """
    def __init__(self,n_draws, mu0, omega0, nu0,
                 sigma_sq0, poly_degree=0, cred_int=95):
        self.w = None
        self.n_draws = n_draws
        self.poly_degree = poly_degree
        self.cred_int = cred_int

        #prior parameters
        self.mu0 = mu0
        self.omega0 = omega0
        self.nu0 = nu0
        self.sigma_sq0 = sigma_sq0

    # allows for simulation from the scaled inverse chi squared
    # distribution. assumes the variance is distributed according to
    # this distribution.
    # reference:
    # https://en.wikipedia.org/wiki/Scaled_inverse_chi-squared_distribution

    def _draw_scaled_inv_chi_sq(self, n, df, scale):
        X = chi2.rvs(size=n, df = df)
        sigma_sq = df * scale/X
        return sigma_sq

    def fit(self, X, y):
        #if polynomial transformation
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)

        n_samples, n_features = np.shape(X)
        X_X = X.T.dot(X)

        #least squares approximate of beta
        beta_hat = np.linalg.pinv(X_X).dot(X.T).dot(y)

        # the posterior parameters can be determined analytically
        # since we assume conjugate priors for the likelihoods.

        #normal prior / likelihood => Normal posterior
        mu_n = np.linalg.pinv(X_X + self.omega0).dot(X_X.dot(beta_hat) + self.omega0.dot(self.mu0))
        omega_n = X_X+ self.omega0
        #scaled inverse chi-squared prior / likelihood => scaled inverse chi-squared posterior
        nu_n = self.nu0 + n_samples
        sigma_sq_n = (1.0/nu_n)*(self.nu0*self.sigma_sq0+
                                 (y.T.dot(y)+self.mu0.T.dot(self.omega0).dot(self.mu0)-mu_n.T.dot(omega_n.dot(mu_n))))

        # simulate parameter values for n_draws
        beta_draws = np.empty((self.n_draws, n_features))
        for i in range(self.n_draws):
            sigma_sq = self._draw_scaled_inv_chi_sq(n=1, df=nu_n, scale=sigma_sq_n)
            beta = multivariate_normal.rvs(size=1, mean=mu_n[:,0],cov=sigma_sq*np.linalg.pinv(omega_n))
            #save parameter draws
            beta_draws[i,:] = beta

        #Select the mean of the simulated variables as the ones used to make predictions
        self.w = np.mean(beta_draws, axis=0)

        # lower and upper boundary of the credible interval
        l_eti = 50 - self.cred_int/2
        u_eti = 50 - self.cred_int/2
        self.eti = np.array([[np.percentile(beta_draws[:,i], q=l_eti),
                              np.percentile(beta_draws[:,i], q=u_eti)] for i in range(n_features)])


    def predict(self, X, eti = False):
        #if polynomial transformation
        if self.poly_degree:
            X = polynomial_features(X, degree = self.poly_degree)

        y_pred = X.dot(self.w)
        # if the lower and upper doundaries for the 95%
        # equal tail interval should be returned
        if eti:
            lower_w = self.eti[:,0]
            upper_w = self.eti[:,1]
            y_lower_pred = X.dot(lower_w)
            y_upper_pred = X.dot(upper_w)
            return y_pred, y_lower_pred, y_upper_pred

        return y_pred 
