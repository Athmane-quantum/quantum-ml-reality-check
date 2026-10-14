import time, numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
rng = np.random.default_rng(0)
X = rng.dirichlet(np.ones(256), 2000); y = rng.uniform(-1,1,2000)
t=time.time()
gs = GridSearchCV(MLPRegressor(max_iter=200, random_state=0), {"hidden_layer_sizes":[(32,),(64,64),(128,128)],"alpha":[1e-4,1e-2,1],"learning_rate_init":[1e-3,1e-2]}, cv=3, n_jobs=1)
import warnings; warnings.filterwarnings("ignore")
gs.fit(X,y); print("one GridSearchCV (18 cfg x 3 folds, m=8): %.1f s" % (time.time()-t))
