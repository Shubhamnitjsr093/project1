import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
import pandas as pd

def evaluate_model_plot(y_true, y_predict, residual=False, show=True):
    """
    Evaluate model error.

    Args:
    y_true: Array-like, true target values.
    y_predict: Array-like, predicted target values.
    residual: Boolean, whether to plot residual histogram.
    show: Boolean, whether to show the plots.

    Returns:
    dict: Dictionary containing error metrics.
    """
    if show:
        print("Start drawing")
        plt.figure(figsize=(7, 5), dpi=400)
        plt.rcParams['font.sans-serif'] = ['Arial']  
        plt.rcParams['axes.unicode_minus'] = False  
        plt.grid(linestyle="--")  
        ax = plt.gca() 
        plt.scatter(y_true, y_predict, color='red')
        plt.plot(y_predict, y_predict, color='blue')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.xlabel("Measured", fontsize=12, fontweight='bold')
        plt.ylabel("Predicted", fontsize=12, fontweight='bold')
        plt.savefig('./genetic.svg', format='svg')
        plt.show()

        if residual:
            plt.figure(figsize=(7, 5), dpi=400)
            plt.hist(np.array(y_true)-np.array(y_predict), 40)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel("Residual", fontsize=20)
            plt.ylabel("Freq", fontsize=20)
            plt.show()

    n = len(y_true)
    MSE = mean_squared_error(y_true, y_predict)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_true, y_predict)
    R2 = r2_score(y_true, y_predict)

    print("Number of samples:", round(n))
    print("Root Mean Squared Error (RMSE):", round(RMSE, 3))
    print("Mean Squared Error (MSE):", round(MSE, 3))
    print("Mean Absolute Error (MAE):", round(MAE, 3))
    print("R-squared (R2):", round(R2, 3))

    return {"n": n, "MSE": MSE, "RMSE": RMSE, "MAE": MAE, "R2": R2}

def draw_corrheatmap(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    """
   Draw a heat map of the correlation coefficient matrix
    """
    dfData = df.corr()
    plt.subplots(figsize=(9, 9)) 
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")

    plt.show()


def cbrt_override(x):
    """ Calculate the cube root"""
    if x>=0:
        return x**(1/3)
    else :
        return -(-x)**(1/3)


def regressorOp(X, Y):
    """
    This will optimize the parameters for the SVR
    X:X_train+X_validation
    Y:Y_train+Y_validation
    """
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    regr_rbf = SVR(kernel="rbf")
   
    C = [pow(10,x) for x in range(-10, 6)]
    gamma =[pow(10,x) for x in range(-2, 3)] 
    epsilon = list(pow(10,x) for x in range(-3,3))
    
    parameters = {"C":C, "gamma":gamma, "epsilon":epsilon}
    
    gs = GridSearchCV(regr_rbf, parameters, scoring="neg_mean_squared_error")
    gs.fit(X, Y)
    
    print ("Best Estimator:\n", gs.best_estimator_)
    print ("Type: ", type(gs.best_estimator_))
    return gs.best_estimator_ 


def read_df(df,target="target",sc=True):
    # import data
    dataset = df
    dataset = dataset.dropna()
    X = dataset.drop([target], axis=1).values
    Y = dataset[target].values
    if sc == True:
        # StandardScaler
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_transform = sc.fit_transform(X)
        result_dict = {"X": X, "Y": Y, "sc": sc, "df": dataset, "X_transform": X_transform}
    else:
	    result_dict = {"X": X, "Y": Y, "df": dataset}    
    return result_dict



def mean_relative_error(y_true, y_pred):
    """calculate MRE"""
    import numpy as np
    relative_error = np.average(np.abs(y_true - y_pred) / y_true, axis=0)
    return relative_error
def Linear_SVR(C=1.0, gamma=0.1, epsilon=1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVR(kernel="linear", C=C, gamma=gamma, epsilon=epsilon))
    ])


def RBF_SVR(C=1.0, gamma=1, epsilon=1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon))
    ])


def Poly_LinearRegression(degree=2):
    return Pipeline([('poly', PolynomialFeatures(degree=degree)),
                     ('linear', LinearRegression())])


def draw_feature_importance(features, feature_importance):
    """
    features: name
    feature_importance:
    """
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(dpi=400)
    plt.barh(pos, list(feature_importance[sorted_idx]), align='center')
    plt.yticks(pos, list(features[sorted_idx]), fontsize=5)
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.show()


def model_fit_evaluation(model, x_train, y_train, x_test, y_test, n_fold=5):
    """Fit the model using cross-validation and evaluate its performance.

    Args:
    clf: Model object implementing 'fit' and 'predict' methods.
    x_train: Training data (features) including validation set.
    y_train: Target values for training data.
    x_test: Test data (features).
    n_fold: Number of folds for cross-validation. Default is 5.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=0)
    print(model)
    result = pd.DataFrame()
    for i, (train_index, test_index) in enumerate(kf.split(range(len(x_train)))):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_validation = x_train[test_index]  # get validation set
        y_validation = y_train[test_index]
        model.fit(x_tr, y_tr)

        result_subset = pd.DataFrame()  # save the prediction
        result_subset["y_validation"] = y_validation
        result_subset["y_pred"] = model.predict(x_validation)
        result = result.append(result_subset)
    print("cross_validation_error in validation set：")
    c = evaluate_model_plot(result["y_validation"], result["y_pred"], show=False)

    print("error in testing set：")
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    error_metric_testing = evaluate_model_plot(y_test, y_test_pred, show=False)  # 不画图
    print("====================================")
    return error_metric_testing


import itertools
import pandas as pd


def back_forward_feature_selection(model, X_train, Y_train, X_validation, Y_validation, metric):
    """X_train X_validation is dataFrame
    metric is evalucation function
    return metric for each step
    metric：The smaller the error measurement function, the better
    """
    # init
    # record result
    result_df = pd.DataFrame()
    features = X_train.columns
    best_score = 1e10
    best_features = features
    features_number = len(best_features)

    for i in range(len(features) - 1):
        # once back and find the best features for this number of features
        best_score = 1e10
        for sub_features in itertools.combinations(best_features, features_number - 1):
            sub_features = list(sub_features)
            model.fit(X_train[sub_features], Y_train)
            score = metric(Y_validation, model.predict(X_validation[sub_features]))
            df_line = pd.DataFrame(
                {"features": [",".join(sub_features)], "metric": score, "n_feature": len(sub_features)})
            result_df = result_df.append(df_line, ignore_index=True)
            if (best_score > score):
                best_score = score
                best_features = sub_features
       
        features_number = len(best_features)
  
    result_df = result_df.sort_values(by="metric", ascending=False)
    return result_df