import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost_learner = AdaBoost(iterations=n_learners, wl=DecisionStump).fit(train_X, train_y)
    partial_test_loss = []
    partial_train_loss = []
    for i in range(1, n_learners + 1):
        partial_test_loss.append(adaboost_learner.partial_loss(test_X, test_y, i))
        partial_train_loss.append(adaboost_learner.partial_loss(train_X, train_y, i))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=partial_test_loss, mode='lines', name='Test Error'))
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=partial_train_loss, mode='lines', name='Train Error'))
    fig.update_layout(title="Adaboost Test and Train error as a function of Number of Fitted Learners",
                      xaxis_title='Number of fitted learners',
                      yaxis_title='Misclassification Error')
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    #
    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.1, vertical_spacing=0.3,
                        subplot_titles=[rf"$\text{{{t} Classifiers}}$" for t in T])
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X: adaboost_learner.partial_predict(X, t), lims[0],
                                         lims[1], showscale=False), go.Scatter(x=test_X[:, 0], y=test_X[:, 1]
                                                                               , mode="markers", showlegend=False,
                                                                               marker=dict(color=test_y,
                                                                                           colorscale=[custom[0],
                                                                                                       custom[-1]],
                                                                                           line=dict(color="black",
                                                                                                     width=1)))],
                       rows=1, cols=i + 1)
    fig.update_layout(height=500, width=2000).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    from IMLearn.metrics import accuracy
    best_accuracy = np.inf
    model_chosen = T[0]
    for i, t in enumerate(T):
        accuracy_best_performance = adaboost_learner.partial_loss(test_X, test_y, t)
        if accuracy_best_performance < best_accuracy:
            best_accuracy = accuracy_best_performance
            model_chosen = t
    best_accuracy = 100-best_accuracy*100
    fig = go.Figure([decision_surface(lambda X: adaboost_learner.partial_predict(X, model_chosen), lims[0],
                                     lims[1], showscale=False), go.Scatter(x=test_X[:, 0], y=test_X[:, 1]
                                                                           , mode="markers", showlegend=False,
                                                                           marker=dict(color=test_y,
                                                                                       colorscale=[custom[0],
                                                                                                   custom[-1]],
                                                                                       line=dict(color="black",
                                                                                                 width=1)))])
    fig.update_layout(title="Highest Accuracy Ensemble - "+str(model_chosen)+""
                            " Classifiers with Accuracy of " + str(best_accuracy)+" %")
    fig.show()
    # Question 4: Decision surface with weighted samples
    fig = go.Figure([decision_surface(adaboost_learner.predict, lims[0],
                                     lims[1], showscale=False), go.Scatter(x=train_X[:, 0], y=train_X[:, 1]
                                    ,mode="markers", showlegend=False,
                                    marker=dict(color=train_y,
                                                colorscale=[custom[0],
                                                custom[-1]],
                                                symbol=np.where(train_y==1,"circle","x"),
                                                line=dict(color="black", width=1),
                                                size=(adaboost_learner.D_/np.max(adaboost_learner.D_))*20))])
    fig.update_layout(title="Decision Surface with Weighted Training Set - With Noise", width=500, height=500)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
