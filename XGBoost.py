from sklearn.model_selection import GridSearchCV
import xgboost as xgb


class XGBoost:
    def __init__(self):
        super().__init__()
        self.model = xgb.XGBClassifier(objective='binary:logistic', max_depth=8, learning_rate=0.001,
                                       n_estimators=150, subsample=0.3, scale_pos_weight=1)
        self.bst = None

    def train(self, x_train, y_train):
        self.bst = self.model.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.model.predict_proba(x_test)[:, 1]
        return y_pred

    def grid_search(self, x_train, y_train):
        classifier = xgb.XGBClassifier(objective='binary:logistic', max_depth=10, learning_rate=0.01,
                                       n_estimators=200, subsample=0.3, scale_pos_weight=1)
        grid_param = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 8, 10, 12, 15],
            'learning_rate': [0.001, 0.01, 0.1],
            'subsample': [0.3, 0.5, 0.7]
        }
        gd_sr = GridSearchCV(estimator=classifier, param_grid=grid_param, scoring='f1', cv=5, verbose=2, n_jobs=-1)
        gd_sr.fit(x_train, y_train)
        best_parameters = gd_sr.best_params_
        print(best_parameters)
        best_result = gd_sr.best_score_
        print(best_result)

    def cross_validation(self, x_train, y_train, params=None):
        if params is None:
            params = {'max_depth': 10, 'learning_rate': 0.01,
                      'objective': 'binary:logistic', 'scale_pos_weight': 1,
                      'n_estimators': 200, 'subsample': 0.3}
        d_train = xgb.DMatrix(x_train, label=y_train)
        xgb_cv = xgb.cv(params, d_train, nfold=10, num_boost_round=1000, metrics=['auc'], early_stopping_rounds=100)
        return xgb_cv.shape[0]  # the best number of rounds

    def get_booster(self):
        return self.bst

    def get_model(self):
        return self.model
