
import os
import psutil
import platform
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack
# import joblib
# from itertools import permutations
from itertools import combinations, chain
import numpy as np
import pandas as pd
import re
import pytz # timezone
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Queue

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
# from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
# from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

!pip install num2words
from num2words import num2words
