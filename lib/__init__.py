import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid


from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, BayesianRidge
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor


