from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import matplotlib.pyplot as plt

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    n_samples = 1000
    x = np.random.normal(10, 1, n_samples)
    univariateGaussian = UnivariateGaussian().fit(x)
    print("(" + str(univariateGaussian.mu_) + " ," + str(univariateGaussian.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    expectation_arr = []
    samples_arr = []

    for n_samples in range(10, 1000, 10):
        univariateGaussian.fit(x[0:n_samples])
        expectation_arr.append(abs(10-univariateGaussian.mu_))
        samples_arr.append(n_samples)
    plt.plot(samples_arr, expectation_arr)
    plt.xlabel("Number of samples")
    plt.ylabel("Expectation error")
    plt.title("Error to Number of Samples")
    plt.show()



    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = np.c_[x, univariateGaussian.pdf(x)]
    pdfs = pdfs[pdfs[:,1].argsort()]
    plt.scatter(pdfs[:,0],pdfs[:,1])
    plt.xlabel("x Values")
    plt.ylabel("Density")
    plt.title("Histogram of Empirical PDF")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    sigma = np.array([[1,0.2,0,0.5],[0.2,2,0,0],[0,0,1,0],[0.5,0,0,1]])
    x = np.random.multivariate_normal(mu, sigma, 1000)
    multivariateGaussian = MultivariateGaussian().fit(x)
    print("The expectation vector is:")
    print(multivariateGaussian.mu_)
    print("The covariance matrix is:")
    print(multivariateGaussian.cov_)

    # Question 5 - Likelihood evaluation
    f_values = np.linspace(-10,10,200)
    log_likelihood = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            mu = np.array([f_values[i], 0, f_values[j],0])
            log_likelihood[i][j] = multivariateGaussian.log_likelihood(mu,sigma,x)

    #Plotting log_likelihood as a function of f1 and f3
    fig, ax = plt.subplots()
    im = ax.imshow(log_likelihood, cmap='viridis')

    #Setting x axis
    x_step = 10
    x_ticks = np.arange(0, len(f_values), x_step)
    x_tick_labels = [f"{f:.3f}" for f in f_values[x_ticks]]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xticks(np.arange(0, len(f_values), x_step))
    ax.set_xticklabels(f_values[::x_step])
    # Setting Y axis
    y_step = 10
    y_ticks = np.arange(0, len(f_values), y_step)
    y_tick_labels = [f"{f:.3f}" for f in f_values[y_ticks]]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    cbar = ax.figure.colorbar(im, ax=ax)
    #Setting titles
    ax.set_title("Heatmap of Log-Likelihood")
    ax.set_xlabel("f3")
    ax.set_ylabel("f1")
    plt.show()

    # Question 6 - Maximum likelihood
    # The maximum likelihood was achieved at approximately (-0.05,3.958)



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
