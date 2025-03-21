from xgboost import XGBClassifier
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from xgboost import dask as dxgb
import dask.array as da
import dask.distributed

def iris_test():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data['data'], 
        data['target'],
        test_size=.2
    )
    
    # create model instance
    bst = XGBClassifier(
        n_estimators=2,
        max_depth=2,
        learning_rate=1,
        objective='binary:logistic'
    )
    # fit model
    bst.fit(X_train, y_train)
    # make predictions
    preds = bst.predict(X_test)
    
    X, y = load_diabetes(return_X_y=True)
    est = HistGradientBoostingRegressor().fit(X, y)
    bsr = XGBRegressor(learning_rate=0.1).fit(X, y)
    return bst, est, bsr

cluster = dask.distributed.LocalCluster()
client = dask.distributed.Client(cluster)
num_obs = 1e5
num_features = 20
X = da.random.random(size=(num_obs, num_features), chunks=(1000, num_features))
y = da.random.random(size=(num_obs, 1), chunks=(1000, 1))


dtrain = dxgb.DaskDMatrix(client, X, y)
output = dxgb.train(
    client, 
    {"verbosity": 2, "tree_method": "hist", "objective": "reg:squarederror"},
    dtrain,
    num_boost_round=4,
    evals=[(dtrain, "train")],
)
