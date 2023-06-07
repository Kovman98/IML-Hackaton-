from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(fit: Perceptron, _, __):
            if X is not None and y is not None:
                loss = fit.loss(X, y)
                losses.append(loss)

        perceptron = Perceptron(callback=callback)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(losses) + 1)), y=losses, mode='lines'))
        fig.update_layout(title="Loss as a function of Fitting Iteration in " + n + " dataset", xaxis_title="Iteration",
                          yaxis_title="Loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        gaussian = GaussianNaiveBayes()
        gaussian.fit(X, y)
        gaussian_prediction = gaussian.predict(X)
        lda = LDA().fit(X, y)
        LDA_prediction = lda.predict(X)

        from IMLearn.metrics import accuracy
        accuracy_gaussian = accuracy(y, gaussian_prediction) * 100
        accuracy_LDA = accuracy(y, LDA_prediction) * 100
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Gaussian Naive Bayes with " + str("{:.2f}".format(accuracy_gaussian))
                                            + "% Accuracy",
                                            "LDA with " + str("{:.2f}".format(accuracy_LDA)) + "% Accuracy"))
        trace1 = go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", name="Gaussian NAive Bayes predictions",
                            marker=dict(color=gaussian_prediction, symbol=y))
        trace2 = go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", name="LDA predictions",
                            marker=dict(color=LDA_prediction, symbol=y))
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=2)
        gaussian_center = gaussian.mu_
        gaussian_covariances = gaussian.vars_
        lda_center = lda.mu_
        lda_covariance = lda.cov_
        fig.add_trace(go.Scatter(x=gaussian_center[:, 0], y=gaussian_center[:, 1],
                                mode="markers", marker=dict(symbol="x", color="black")), row=1, col=1)
        fig.add_trace(go.Scatter(x=lda_center[:, 0], y=lda_center[:, 1],
                                 mode="markers", marker=dict(symbol="x", color="black")), row=1, col=2)
        for i in range(3):
            fig.add_trace(get_ellipse(gaussian_center[i], np.diag(gaussian_covariances[i])), row=1, col=1)
            fig.add_trace(get_ellipse(lda_center[i], lda_covariance), row=1, col=2)
        fig.update_layout(title="Gaussian Dataset: "+f)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
