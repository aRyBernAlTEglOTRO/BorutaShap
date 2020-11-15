from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from borutashap import BorutaShap, load_data


def test_models(data_type, models):
    x, y = load_data(data_type=data_type)

    for key, value in models.items():
        print("Testing: " + str(key))
        # no model selected default is Random Forest, if classification is False it is a Regression problem
        Feature_Selector = BorutaShap(
            model=value, importance_measure="shap", classification=True
        )

        Feature_Selector.fit(
            x=x, y=y, n_trials=5, random_state=0, train_or_test="train"
        )

        # Returns Boxplot of features disaplay False or True to see the plots for automation False
        Feature_Selector.plot(
            x_size=12,
            figsize=(12, 8),
            y_scale="log",
            which_features="all",
            display=False,
        )


if __name__ == "__main__":
    tree_classifiers = {
        "tree-classifier": DecisionTreeClassifier(),
        "forest-classifier": RandomForestClassifier(),
        "xgboost-classifier": XGBClassifier(),
        "lightgbm-classifier": LGBMClassifier(),
        "catboost-classifier": CatBoostClassifier(),
    }

    tree_regressors = {
        "tree-regressor": DecisionTreeRegressor(),
        "forest-regressor": RandomForestRegressor(),
        "xgboost-regressor": XGBRegressor(),
        "lightgbm-regressor": LGBMRegressor(),
        "catboost-regressor": CatBoostRegressor(),
    }

    test_models("regression", tree_regressors)
    test_models("classification", tree_classifiers)
