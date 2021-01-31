import os
from sklearn import preprocessing
from sklearn import ensemble
import pandas as pd
from sklearn import metrics
import joblib

from . import dispatcher


TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

if __name__=="__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    ytrain = train_df.digit.values
    yvalid = valid_df.digit.values

    train_df = train_df.drop(["digit", "kfold"], axis = 1)
    valid_df = valid_df.drop(["digit", "kfold"], axis = 1)

    valid_df = valid_df[train_df.columns]

#Data is ready to be trained
print(MODEL)
clf = dispatcher.MODELS[MODEL]
clf.fit(train_df, ytrain)
preds = clf.predict(valid_df)
print(metrics.accuracy_score(yvalid, preds))

joblib.dump(clf, f"models/{MODEL}_fold_{FOLD}.pkl")
