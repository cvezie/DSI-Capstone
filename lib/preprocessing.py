import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES

from tqdm import tqdm

def display_segments(dataframe, pca, n_components, n_offset=0):

    pca_components = pca.components_[n_offset:n_components+n_offset]
    exp_var_rat = np.append(pca.explained_variance_ratio_[n_offset:n_components+n_offset], 0)
    # Dimension indexing

    dimensions = ['Dimension {}'.format(i+n_offset) for i in range(1,len(pca_components)+1)] + ['']

    # PCA components
    components = pd.DataFrame(np.round(pca_components, 4), columns = dataframe.columns)
    components = components.append(0*components.loc[0])
    components.index = dimensions

    # PCA explained variance
    ratios = exp_var_rat.reshape(len(pca_components)+1, 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (20,6))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar')
    plt.legend(loc=1)
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios
    for i, ev in enumerate(exp_var_rat[:n_components]):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

def cluster_range(X, range_n_clusters):

    for n_clusters in tqdm(range_n_clusters):
        # Create a subplot with 1 row and 2 columns
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.set_size_inches(15, 15)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax3.scatter(X[:, 0], X[:, 2], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax3.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax3.set_title("The visualization of the clustered data.")
        ax3.set_xlabel("Feature space for the 1st feature")
        ax3.set_ylabel("Feature space for the 3rd feature")

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax4.scatter(X[:, 1], X[:, 2], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax4.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax4.set_title("The visualization of the clustered data.")
        ax4.set_xlabel("Feature space for the 1st feature")
        ax4.set_ylabel("Feature space for the 3rd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    return fig
        
    
    
def _transform_selected(X, transform, selected="all", copy=True,
                        retain_ordering=False):
    """Apply a transform function to portion of selected features
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Dense array or sparse matrix.
    transform : callable
        A callable transform(X) -> X_transformed
    copy : boolean, optional
        Copy X even if it could be avoided.
    selected : "all" or array of indices or mask
        Specify which features to apply the transform to.
    retain_ordering : boolean, default False
        Specify whether the initial order of features has
        to be maintained in the output
    Returns
    -------
    X : array or sparse matrix, shape=(n_samples, n_features_new)
    """
    X = check_array(X, accept_sparse='csc', copy=copy, dtype=FLOAT_DTYPES)

    if isinstance(selected, six.string_types) and selected == "all":
        return transform(X)

    if len(selected) == 0:
        return X

    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    not_sel = np.logical_not(sel)
    n_selected = np.sum(sel)

    if n_selected == 0:
        # No features selected.
        return X
    elif n_selected == n_features:
        # All features selected.
        return transform(X)
    else:
        X_sel = transform(X[:, ind[sel]])
        X_not_sel = X[:, ind[not_sel]]

        if retain_ordering:
            # As of now, X is expected to be dense array
            X[:, ind[sel]] = X_sel
            return X
        if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
            return sparse.hstack((X_sel, X_not_sel))
        else:
            return np.hstack((X_sel, X_not_sel))

def _boxcox(X, i, lambda_x=None):
    x = X[:, i]
    if lambda_x is None:
        x, lambda_x = stats.boxcox(x, lambda_x)
        return x, lambda_x
    else:
        x = stats.boxcox(x, lambda_x)
        return x


def boxcox(X, copy=True):
    """BoxCox transform to the input data
    Apply boxcox transform on individual features with lambda
    that maximizes the log-likelihood function for each feature
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be transformed. Should contain only positive data.
    copy : boolean, optional, default=True
        Set to False to perform inplace transformation and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).
    Returns
    -------
    X_tr : array-like, shape (n_samples, n_features)
        The transformed data.
    References
    ----------
    G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal of the
    Royal Statistical Society B, 26, 211-252 (1964).
    """
    X = check_array(X, ensure_2d=True, dtype=FLOAT_DTYPES, copy=copy)
    if np.any(X <= 0):
        raise ValueError("BoxCox transform can only be applied "
                         "on positive data")
    n_features = X.shape[1]
    outputs = Parallel(n_jobs=-1)(delayed(_boxcox)(X, i, lambda_x=None)
                                  for i in range(n_features))
    output = np.concatenate([o[0][..., np.newaxis] for o in outputs], axis=1)
    return output


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """BoxCox transformation on individual features.
    Boxcox transform wil be applied on each feature (each column of
    the data matrix) with lambda evaluated to maximise the log-likelihood
    Parameters
    ----------
    transformed_features : "all" or array of indices or mask
        Specify what features are to be transformed.
        - "all" (default): All features are to be transformed.
        - array of int: Array of feature indices to be transformed..
        - mask: Array of length n_features and with dtype=bool.
    copy : boolean, optional, default=True
        Set to False to perform inplace computation.
    Attributes
    ----------
    transformed_features_ : array of int
        The indices of the features to be transformed
    lambdas_ : array of float, shape (n_transformed_features,)
        The parameters of the BoxCox transform for the selected features.
    n_features_ : int
        Number of features in input during fit
    Notes
    -----
    The Box-Cox transform is given by::
        y = (x ** lmbda - 1.) / lmbda,  for lmbda > 0
            log(x),                     for lmbda = 0
    ``boxcox`` requires the input data to be positive.
    """
    def __init__(self, transformed_features="all", n_jobs=1, copy=True):
        self.transformed_features = transformed_features
        self.n_jobs = n_jobs
        self.copy = copy

    def fit(self, X, y=None):
        """Estimate lambda for each feature to maximise log-likelihood
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to fit by apply boxcox transform,
            to each of the features and learn the lambda.
        y : ignored
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, ensure_2d=True, dtype=FLOAT_DTYPES)
        self.n_features_ = X.shape[1]
        if self.transformed_features is "all":
            self.transformed_features_ = np.arange(self.n_features_)
        else:
            self.transformed_features_ = np.copy(self.transformed_features)
            if self.transformed_features_.dtype == np.bool:
                self.transformed_features_ = \
                    np.where(self.transformed_features_)[0]
        if np.any(X[:, self.transformed_features_] <= 0):
            raise ValueError("BoxCox transform can only be applied "
                             "on positive data")
        out = Parallel(n_jobs=self.n_jobs)(delayed(_boxcox)(X, i,
                                           lambda_x=None)
                                           for i in self.transformed_features_)
        self.lambdas_ = np.array([o[1] for o in out])
        return self

    def transform(self, X):
        """Transform each feature using the lambdas evaluated during fit time
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to apply boxcox transform.
        Returns
        -------
        X_tr : array-like, shape (n_samples, n_features)
            The transformed data.
        """
        X = check_array(X, ensure_2d=True, dtype=FLOAT_DTYPES, copy=self.copy)
        if any(np.any(X[:, self.transformed_features_] <= 0, axis=0)):
            raise ValueError("BoxCox transform can only be applied "
                             "on positive data")
        if X.shape[1] != self.n_features_:
            raise ValueError("X has a different shape than during fitting.")
        X_tr = _transform_selected(X, self._transform,
                                   self.transformed_features_,
                                   copy=False, retain_ordering=True)
        return X_tr

    def _transform(self, X):
        outputs = Parallel(n_jobs=self.n_jobs)(
            delayed(_boxcox)(X, i, self.lambdas_[i])
            for i in range(len(self.transformed_features_)))
        output = np.concatenate([o[..., np.newaxis] for o in outputs], axis=1)
        return output

    def inverse_transform(self, X):
        """Inverse transform each feature using the lambdas evaluated during fit time
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The transformed data after boxcox transform.
        Returns
        -------
        X_inv : array-like, shape (n_samples, n_features)
            The original data.
        Notes
        -----
        The inverse Box-Cox transform is given by::
        y = log(x * lmbda + 1.) / lmbda,  for lmbda > 0
            exp(x),                       for lmbda = 0
        """

        X = check_array(X, ensure_2d=True, dtype=FLOAT_DTYPES, copy=self.copy)
        if X.shape[1] != self.n_features_:
            raise ValueError("X has a different shape than during fitting.")
        X_inv = _transform_selected(X, self._inverse_transform,
                                    self.transformed_features_, copy=False,
                                    retain_ordering=True)
        return X_inv

    def _inverse_transform(self, X):
        X_inv = X.copy()

        mask = self.lambdas_ != 0
        mask_lambdas = self.lambdas_[mask]
        Xinv_mask = X_inv[:, mask]
        Xinv_mask *= mask_lambdas
        np.log1p(Xinv_mask, out=Xinv_mask)
        Xinv_mask /= mask_lambdas
        np.exp(Xinv_mask, out=Xinv_mask)
        X_inv[:, mask] = Xinv_mask

        mask = self.lambdas_ == 0
        X_inv[:, mask] = np.exp(X_inv[:, mask])
        return X_inv