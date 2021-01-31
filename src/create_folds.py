import pandas as pd
from sklearn import model_selection
from sklearn import datasets

def create_folds(df):
    df['kfold'] = -1
    df = df.sample(frac = 1).reset_index(drop = True)
    kf = model_selection.KFold(n_splits = 5, shuffle = False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X = df, y = df.digit.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx,'kfold'] = fold

    df.to_csv("input/train_folds.csv", index = False)

if __name__ == "__main__":
    print("Code start")
    df,y = datasets.fetch_openml('mnist_784',
                                version = 1,
                                return_X_y = True)
    df['digit'] = y
    create_folds(df)
    




    
    