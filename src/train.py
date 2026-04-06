import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import pickle
import joblib 
from evaluate import evaluate_model





def train(X_train, y_train, X_test, y_test, X_val, y_val):

    # Train Baseline LR Model # LR baseline requires imputer for serving - not production-ready.
    imputer = SimpleImputer(strategy="median")
    X_train_lr = imputer.fit_transform(X_train)
    X_train_lr = pd.DataFrame(X_train_lr, columns=X_train.columns, index=X_train.index)
    X_test_lr = imputer.transform(X_test)
    X_test_lr = pd.DataFrame(X_test_lr, columns=X_test.columns, index=X_test.index)

    lrbaselinemodel = LogisticRegression(class_weight='balanced', max_iter=1000, solver = 'saga',random_state = 42)
    lrbaselinemodel.fit(X_train_lr,y_train)
    # Train XGBoost
    xgboostmodel = XGBClassifier(scale_pos_weight = 27.6, objective = "binary:logistic", eval_metric = "aucpr", max_depth = 6, n_estimators = 500, learning_rate = 0.1, early_stopping_rounds=10, seed = 42)
    xgboostmodel.fit(X_train, y_train, eval_set=[(X_val, y_val)])


    #evaluate the model
    evalres = evaluate_model(xgboostmodel, X_test, y_test, "XGB")





if __name__ == "__main__":
    with open('X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    with open('X_val.pkl', 'rb') as f:
        X_val = pickle.load(f)
    with open('y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)

    train(X_train, y_train, X_test, y_test, X_val, y_val)

    #Your current results:
    # XGBoost AUCPR: 0.692 — solid first run
    # LR: failed to converge — expected given 487 features, 472k rows, imbalanced classes

