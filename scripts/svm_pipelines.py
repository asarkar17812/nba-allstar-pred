import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def get_base_pipeline(cfg, c_val=1.0, weights={0: 1, 1: 3}):
    """
    BASELINE: Data is already scaled. 
    Just feeds the pre-scaled X into the SVM.
    """
    return Pipeline([
        ('scaler', StandardScaler()), 
        ('svm', SVC(
            C=c_val, 
            kernel='linear', 
            probability=cfg.PROBABILITY,
            max_iter=cfg.MAX_ITER, 
            tol=cfg.TOL,
            random_state=cfg.SEED,
            class_weight = weights
        ))
    ])

def get_poly_pipeline(cfg, c_val=0.1, weights={0: 1, 1: 3}):
    """
    TRANSFORM 1: Degree 2 Interactions.
    Scales the interactions AFTER they are created to ensure 
    the SVM doesn't 'choke' on the new numerical ranges.
    """
    return Pipeline([
        ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
        ('scaler', StandardScaler()), 
        ('svm', SVC(
            C=c_val, 
            kernel='linear', 
            probability=cfg.PROBABILITY,
            max_iter=cfg.MAX_ITER, 
            tol=cfg.TOL,
            random_state=cfg.SEED,
            class_weight = weights
        ))
    ])

def get_kmeans_pipeline(n_clusters=10, c_val=1.0, weights={0:1, 1:3}):
    """
    TRANSFORM: K-Means Cluster Distances.
    Replaces raw stats with distances to N player archetypes.
    """
    # We use a custom wrapper or a simple FunctionTransformer to get distances
    return Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, n_init=10)),
        ('svm', LinearSVC(C=c_val, class_weight=weights, dual=False, max_iter=20000))
    ])

def get_pca_pipeline(n_components=5, c_val=1.0, weights={0:1, 1:3}):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('svm', LinearSVC(C=c_val, class_weight=weights, dual=False, max_iter=20000))    
])