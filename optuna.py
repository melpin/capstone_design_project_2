import optuna

import numpy as np
import optuna.integration.lightgbm as lgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import ember


def objective(trial):
    X, y = ember.read_vectorized_features('./sample/merge', 20000, 3154)

    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.4, random_state=777)
    valid_x, test_x, valid_y, test_y = train_test_split(val_x, val_y, test_size=0.5, random_state=777)
    sc = StandardScaler()
    train_x = sc.fit_transform(train_x)
    valid_x = sc.transform(valid_x)
    test_x = sc.transform(test_x)

    train_data_set = lgb.Dataset(train_x, train_y)
    valid_data_sets = lgb.Dataset(valid_x, valid_y)

    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        "verbosity": -1,
        "boosting_type": "gbdt",
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        # 'num_leaves': 2048,  # 전체 트리의 leave 수, 디폴트값 31
        # 'max_depth': 16,  # 트리 최대 깊이
        # 'min_data_in_leaf': 1000,  # 리프가 갖는 최소한의 레코드, 디폴트값은  20으로 최적의 값
        # 'num_iterations': 1000,  # 1000 -> 1500
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }

    gbm = lgb.train(param, train_data_set, valid_sets=[valid_data_sets], verbose_eval=False)
    pred_y = gbm.predict(test_x)
    y_pred = np.where(np.array(pred_y) > 0.7, 1, 0)
    accuracy = sklearn.metrics.accuracy_score(test_y, y_pred)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
