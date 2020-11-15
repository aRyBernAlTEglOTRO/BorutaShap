import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from numpy.random import choice
from scipy.stats import binom_test, ks_2samp
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")


class BorutaShap:

    """
    BorutaShap is a wrapper feature selection method built on the foundations of both the SHAP and Boruta algorithms.

    """

    def __init__(
        self,
        model=None,
        importance_measure="Shap",
        classification=True,
        percentile=100,
        pvalue=0.05,
    ):
        """
        [summary]

        Parameters
        ----------
        model : [type], optional
            [description], by default None
        importance_measure : str, optional
            [description], by default "Shap"
        classification : bool, optional
            [description], by default True
        percentile : int, optional
            [description], by default 100
        pvalue : float, optional
            [description], by default 0.05
        """
        self.importance_measure = importance_measure.lower()
        self.percentile = percentile
        self.pvalue = pvalue
        self.classification = classification
        self.model = model
        self.check_model()

    def check_model(self):
        """
        [summary]

        Raises
        ------
        AttributeError
            [description]
        AttributeError
            [description]
        """
        check_fit = hasattr(self.model, "fit")
        check_predict_proba = hasattr(self.model, "predict")

        try:
            check_feature_importance = hasattr(self.model, "feature_importances_")
        except Exception as e:
            print(f"Meet error when check model: {e}")
            check_feature_importance = True

        if self.model is None:
            if self.classification:
                self.model = RandomForestClassifier()
            else:
                self.model = RandomForestRegressor()
        elif not (check_fit or check_predict_proba):
            raise AttributeError(
                "Model must contain both the fit() and predict() methods"
            )
        elif not check_feature_importance and self.importance_measure == "gini":
            raise AttributeError(
                "Model must contain the feature_importances_ method to use Gini try Shap instead"
            )

    def check_x(self):
        """
        [summary]

        Raises
        ------
        AttributeError
            [description]
        """
        if not isinstance(self.x, pd.DataFrame):
            raise AttributeError("Input data must be a pandas Dataframe")

    def missing_values_y(self):
        """
        [summary]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        AttributeError
            [description]
        """

        if isinstance(self.y, pd.Series):
            return self.y.isnull().any().any()
        elif isinstance(self.y, np.ndarray):
            return np.isnan(self.y).any()
        else:
            raise AttributeError("Y must be a pandas Dataframe or a numpy array")

    def check_missing_values(self):
        """
        [summary]

        Raises
        ------
        ValueError
            [description]
        """
        no_check_models = ("xgb", "catboost", "lgbm")
        model_name = str(type(self.model)).lower()

        if model_name not in no_check_models:
            x_missing = self.x.isnull().any().any()
            y_missing = self.missing_values_y()

            if x_missing or y_missing:
                raise ValueError("There are missing values in your Data")

    def check_if_chose_train_or_test_and_train_model(self):
        """
        [summary]

        Raises
        ------
        ValueError
            [description]
        """
        if self.train_or_test.lower() == "test":
            # keeping the same naming convenetion as to not add complexit later on
            (
                self.x_boruta_train,
                self.x_boruta_test,
                self.y_train,
                self.y_test,
            ) = train_test_split(
                self.x_boruta, self.y, test_size=0.3, random_state=self.random_state
            )
            self.train_model(self.x_boruta_train, self.y_train)
        elif self.train_or_test.lower() == "train":
            # model will be trained and evaluated on the same data
            self.train_model(self.x_boruta, self.y)
        else:
            raise ValueError(
                'The train_or_test parameter can only be "train" or "test"'
            )

    def train_model(self, x, y):
        """
        [summary]

        Parameters
        ----------
        x : [type]
            [description]
        y : [type]
            [description]
        """
        model_name = str(type(self.model)).lower()
        if "catboost" in model_name:
            self.model.fit(x, y, cat_features=self.x_categorical, verbose=False)
        else:
            try:
                self.model.fit(x, y, verbose=False)
            except:
                self.model.fit(x, y)

    def fit(
        self,
        x,
        y,
        n_trials=20,
        random_state=0,
        sample=False,
        train_or_test="test",
        normalize=True,
        verbose=True,
    ):
        """
        [summary]

        Parameters
        ----------
        x : [type]
            [description]
        y : [type]
            [description]
        n_trials : int, optional
            [description], by default 20
        random_state : int, optional
            [description], by default 0
        sample : bool, optional
            [description], by default False
        train_or_test : str, optional
            [description], by default "test"
        normalize : bool, optional
            [description], by default True
        verbose : bool, optional
            [description], by default True
        """
        np.random.seed(random_state)
        self.starting_x = x.copy()
        self.x = x.copy()
        self.y = y.copy()
        self.n_trials = n_trials
        self.random_state = random_state
        self.ncols = self.x.shape[1]
        self.all_columns = self.x.columns.to_numpy()
        self.rejected_columns = []
        self.accepted_columns = []

        self.check_x()
        self.check_missing_values()
        self.sample = sample
        self.train_or_test = train_or_test

        self.features_to_remove = []
        self.hits = np.zeros(self.ncols)
        self.order = self.create_mapping_between_cols_and_indices()
        self.create_importance_history()

        if self.sample:
            self.preds = self.isolation_forest(self.x)

        for trial in tqdm(range(self.n_trials)):
            self.remove_features_if_rejected()
            self.columns = self.x.columns.to_numpy()
            self.create_shadow_features()

            # early stopping
            if self.x.shape[1] == 0:
                break
            else:
                self.check_if_chose_train_or_test_and_train_model()

                (
                    self.x_feature_import,
                    self.shadow_feature_import,
                ) = self.feature_importance(normalize=normalize)

                self.update_importance_history()
                self.hits += self.calculate_hits()
                self.test_features(iteration=trial + 1)

        self.store_feature_importance()
        self.calculate_rejected_accepted_tentative(verbose=verbose)

    def calculate_rejected_accepted_tentative(self, verbose):
        """
        [summary]

        Parameters
        ----------
        verbose : [type]
            [description]
        """
        self.rejected = list(
            set(self.flatten_list(self.rejected_columns))
            - set(self.flatten_list(self.accepted_columns))
        )
        self.accepted = list(set(self.flatten_list(self.accepted_columns)))
        self.tentative = list(
            set(self.all_columns) - set(self.rejected + self.accepted)
        )

        if verbose:
            print(
                str(len(self.accepted))
                + " attributes confirmed important: "
                + str(self.accepted)
            )
            print(
                str(len(self.rejected))
                + " attributes confirmed unimportant: "
                + str(self.rejected)
            )
            print(
                str(len(self.tentative))
                + " tentative attributes remains: "
                + str(self.tentative)
            )

    def create_importance_history(self):
        """
        [summary]
        """
        self.history_shadow = np.zeros(self.ncols)
        self.history_x = np.zeros(self.ncols)

    def update_importance_history(self):
        """
        [summary]
        """
        padded_history_shadow = np.full((self.ncols), np.NaN)
        padded_history_x = np.full((self.ncols), np.NaN)

        for (index, col) in enumerate(self.columns):
            map_index = self.order[col]
            padded_history_shadow[map_index] = self.shadow_feature_import[index]
            padded_history_x[map_index] = self.x_feature_import[index]

        self.history_shadow = np.vstack((self.history_shadow, padded_history_shadow))
        self.history_x = np.vstack((self.history_x, padded_history_x))

    def store_feature_importance(self):
        """
        [summary]
        """
        self.history_x = pd.DataFrame(data=self.history_x, columns=self.all_columns)

        self.history_x["max_shadow"] = [max(i) for i in self.history_shadow]
        self.history_x["min_shadow"] = [min(i) for i in self.history_shadow]
        self.history_x["mean_shadow"] = [np.nanmean(i) for i in self.history_shadow]
        self.history_x["median_shadow"] = [np.nanmedian(i) for i in self.history_shadow]
        self.history_x.dropna(axis=0, inplace=True)

    def results_to_csv(self, filename="feature_importance"):
        """
        [summary]

        Parameters
        ----------
        filename : str, optional
            [description], by default "feature_importance"
        """
        features = pd.DataFrame(
            data={
                "Features": self.history_x.iloc[1:].columns.values,
                "Average Feature Importance": self.history_x.iloc[1:]
                .mean(axis=0)
                .values,
                "Standard Deviation Importance": self.history_x.iloc[1:]
                .std(axis=0)
                .values,
            }
        )

        decision_mapper = self.create_mapping_of_features_to_attribute(
            maps=["Tentative", "Rejected", "Accepted", "Shadow"]
        )
        features["Decision"] = features["Features"].map(decision_mapper)
        features = features.sort_values(
            by="Average Feature Importance", ascending=False
        )

        features.to_csv(filename + ".csv", index=False)

    def remove_features_if_rejected(self):
        """
        [summary]
        """
        if len(self.features_to_remove):
            for feature in self.features_to_remove:
                try:
                    self.X.drop(feature, axis=1, inplace=True)
                except:
                    pass

    @staticmethod
    def average_of_list(lst):
        """
        [summary]

        Parameters
        ----------
        lst : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        return sum(lst) / len(lst)

    @staticmethod
    def flatten_list(array):
        """
        [summary]

        Parameters
        ----------
        array : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        return [item for sublist in array for item in sublist]

    def create_mapping_between_cols_and_indices(self):
        """
        [summary]

        Returns
        -------
        [type]
            [description]
        """
        return dict(zip(self.x.columns.to_list(), np.arange(self.x.shape[1])))

    def calculate_hits(self):
        """
        [summary]

        Returns
        -------
        [type]
            [description]
        """
        shadow_threshold = np.percentile(self.shadow_feature_import, self.percentile)

        padded_hits = np.zeros(self.ncols)
        hits = self.x_feature_import > shadow_threshold

        for (index, col) in enumerate(self.columns):
            map_index = self.order[col]
            padded_hits[map_index] += hits[index]

        return padded_hits

    def create_shadow_features(self):
        """
        [summary]
        """
        self.x_shadow = self.x.apply(np.random.permutation)
        self.x_shadow.columns = ["shadow_" + feature for feature in self.x.columns]
        self.x_boruta = pd.concat([self.x, self.x_shadow], axis=1)

        col_types = self.x_boruta.dtypes
        self.x_categorical = list(
            col_types[(col_types == "category") | (col_types == "object")].index
        )

    @staticmethod
    def calculate_zscore(array):
        """
        [summary]

        Parameters
        ----------
        array : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        mean_value = np.mean(array)
        std_value = np.std(array)
        return [(element - mean_value) / std_value for element in array]

    def feature_importance(self, normalize):
        """
        [summary]

        Parameters
        ----------
        normalize : [type]
            [description]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        ValueError
            [description]
        """
        if self.importance_measure == "shap":
            self.explain()
            vals = self.shap_values

            if normalize:
                vals = self.calculate_zscore(vals)

            x_feature_import = vals[: len(self.x.columns)]
            shadow_feature_import = vals[len(self.x_shadow.columns) :]

        elif self.importance_measure == "gini":
            feature_importances_ = np.abs(self.model.feature_importances_)

            if normalize:
                feature_importances_ = self.calculate_zscore(feature_importances_)

            x_feature_import = feature_importances_[: len(self.x.columns)]
            shadow_feature_import = feature_importances_[len(self.x.columns) :]

        else:
            raise ValueError(
                "No Importance_measure was specified select one of (shap, gini)"
            )

        return x_feature_import, shadow_feature_import

    @staticmethod
    def isolation_forest(x):
        """
        [summary]

        Parameters
        ----------
        x : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        clf = IsolationForest().fit(x)
        preds = clf.score_samples(x)
        return preds

    @staticmethod
    def get_5_percent(num):
        """
        [summary]

        Parameters
        ----------
        num : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        return round(5 / 100 * num)

    def get_5_percent_splits(self, length):
        """
        [summary]

        Parameters
        ----------
        length : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        five_percent = self.get_5_percent(length)
        return np.arange(five_percent, length, five_percent)

    def find_sample(self):
        """
        [summary]

        Returns
        -------
        [type]
            [description]
        """
        loop = True
        iteration = 0
        size = self.get_5_percent_splits(self.x.shape[0])
        element = 1
        sample_indices = None

        while loop:
            sample_indices = choice(
                np.arange(self.preds.size), size=size[element], replace=False
            )
            sample = np.take(self.preds, sample_indices)
            if ks_2samp(self.preds, sample).pvalue > 0.95:
                break

            if iteration == 20:
                element += 1
                iteration = 0

        return self.x_boruta.iloc[sample_indices].copy()

    def explain(self):
        """
        [summary]
        """
        explainer = shap.TreeExplainer(
            self.model, feature_perturbation="tree_path_dependent"
        )

        if self.sample:
            if self.classification:
                # for some reason shap returns values wraped in a list of length 1
                self.shap_values = np.array(explainer.shap_values(self.find_sample()))
                if isinstance(self.shap_values, list):
                    class_inds = range(len(self.shap_values))
                    shap_imp = np.zeros(self.shap_values[0].shape[1])
                    for ind in class_inds:
                        shap_imp += np.abs(self.shap_values[ind]).mean(0)
                    self.shap_values /= len(self.shap_values)
                elif len(self.shap_values.shape) == 3:
                    self.shap_values = np.abs(self.shap_values).sum(axis=0)
                    self.shap_values = self.shap_values.mean(0)
                else:
                    self.shap_values = np.abs(self.shap_values).mean(0)
            else:
                self.shap_values = explainer.shap_values(self.find_sample())
                self.shap_values = np.abs(self.shap_values).mean(0)
        else:
            if self.classification:
                # for some reason shap returns values wraped in a list of length 1
                self.shap_values = np.array(explainer.shap_values(self.x_boruta))
                if isinstance(self.shap_values, list):
                    class_inds = range(len(self.shap_values))
                    shap_imp = np.zeros(self.shap_values[0].shape[1])

                    for ind in class_inds:
                        shap_imp += np.abs(self.shap_values[ind]).mean(0)

                    self.shap_values /= len(self.shap_values)

                elif len(self.shap_values.shape) == 3:
                    self.shap_values = np.abs(self.shap_values).sum(axis=0)
                    self.shap_values = self.shap_values.mean(0)
                else:
                    self.shap_values = np.abs(self.shap_values).mean(0)
            else:
                self.shap_values = explainer.shap_values(self.x_boruta)
                self.shap_values = np.abs(self.shap_values).mean(0)

    @staticmethod
    def binomial_h0_test(array, n, p, alternative):
        """
        [summary]

        Parameters
        ----------
        array : [type]
            [description]
        n : [type]
            [description]
        p : [type]
            [description]
        alternative : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        return [binom_test(x, n=n, p=p, alternative=alternative) for x in array]

    @staticmethod
    def symetric_difference_between_two_arrays(array_one, array_two):
        """
        [summary]

        Parameters
        ----------
        array_one : [type]
            [description]
        array_two : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        set_one = set(array_one)
        set_two = set(array_two)
        return np.array(list(set_one.symmetric_difference(set_two)))

    @staticmethod
    def find_index_of_true_in_array(array):
        """
        [summary]

        Parameters
        ----------
        array : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        length = len(array)
        return list(filter(lambda x: array[x], range(length)))

    @staticmethod
    def bonferoni_corrections(pvals, alpha=0.05, n_tests=None):
        """
        [summary]

        Parameters
        ----------
        pvals : [type]
            [description]
        alpha : float, optional
            [description], by default 0.05
        n_tests : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        pvals = np.array(pvals)

        if not n_tests:
            n_tests = len(pvals)

        alphacBon = alpha / float(n_tests)
        reject = pvals <= alphacBon
        pvals_corrected = pvals * float(n_tests)

        return reject, pvals_corrected

    def test_features(self, iteration):
        """
        [summary]

        Parameters
        ----------
        iteration : [type]
            [description]
        """
        acceptance_p_values = self.binomial_h0_test(
            self.hits, n=iteration, p=0.5, alternative="greater"
        )

        regect_p_values = self.binomial_h0_test(
            self.hits, n=iteration, p=0.5, alternative="less"
        )

        # [1] as function returns a tuple
        modified_acceptance_p_values = self.bonferoni_corrections(
            acceptance_p_values, alpha=0.05, n_tests=len(self.columns)
        )[1]

        modified_regect_p_values = self.bonferoni_corrections(
            regect_p_values, alpha=0.05, n_tests=len(self.columns)
        )[1]

        # Take the inverse as we want true to keep featrues
        rejected_columns = np.array(modified_regect_p_values) < self.pvalue
        accepted_columns = np.array(modified_acceptance_p_values) < self.pvalue

        rejected_indices = self.find_index_of_true_in_array(rejected_columns)
        accepted_indices = self.find_index_of_true_in_array(accepted_columns)

        rejected_features = self.all_columns[rejected_indices]
        accepted_features = self.all_columns[accepted_indices]

        self.features_to_remove = rejected_features

        self.rejected_columns.append(rejected_features)
        self.accepted_columns.append(accepted_features)

    def TentativeRoughFix(self):
        """
        [summary]
        """
        median_tentaive_values = self.history_x[self.tentative].median(axis=0).values
        median_max_shadow = self.history_x["max_Shadow"].median(axis=0)

        filtered = median_tentaive_values > median_max_shadow

        self.tentative = np.array(self.tentative)
        newly_accepted = self.tentative[filtered]

        if len(newly_accepted) < 1:
            newly_rejected = self.tentative
        else:
            newly_rejected = self.symetric_difference_between_two_arrays(
                newly_accepted, self.tentative
            )

        print(
            str(len(newly_accepted))
            + " tentative features are now accepted: "
            + str(newly_accepted)
        )
        print(
            str(len(newly_rejected))
            + " tentative features are now rejected: "
            + str(newly_rejected)
        )

        self.rejected = self.rejected + newly_rejected.tolist()
        self.accepted = self.accepted + newly_accepted.tolist()

    def subset(self, tentative=False):
        """
        [summary]

        Parameters
        ----------
        tentative : bool, optional
            [description], by default False

        Returns
        -------
        [type]
            [description]
        """
        if tentative:
            return self.starting_x[self.accepted + self.tentative.tolist()]
        else:
            return self.starting_x[self.accepted]

    @staticmethod
    def create_list(array, color):
        """
        [summary]

        Parameters
        ----------
        array : [type]
            [description]
        color : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        colors = [color for _ in range(len(array))]
        return colors

    @staticmethod
    def filter_data(data, column, value):
        """
        [summary]

        Parameters
        ----------
        data : [type]
            [description]
        column : [type]
            [description]
        value : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        data = data.copy()
        return data.loc[(data[column] == value) | (data[column] == "Shadow")]

    @staticmethod
    def has_numbers(input_string):
        """
        [summary]

        Parameters
        ----------
        input_string : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        return any(char.isdigit() for char in input_string)

    @staticmethod
    def check_if_which_features_is_correct(my_string):
        """
        [summary]

        Parameters
        ----------
        my_string : [type]
            [description]

        Raises
        ------
        ValueError
            [description]
        """
        my_string = str(my_string).lower()
        error_message = f"{my_string} is not a valid value did you mean to type 'all', 'tentative', 'accepted' or 'rejected' ?"
        assert my_string in ["tentative", "rejected", "accepted", "all"], error_message

    def plot(
        self,
        x_rotation=90,
        x_size=8,
        figsize=(12, 8),
        y_scale="log",
        which_features="all",
        display=True,
    ):
        """
        [summary]

        Parameters
        ----------
        x_rotation : int, optional
            [description], by default 90
        x_size : int, optional
            [description], by default 8
        figsize : tuple, optional
            [description], by default (12, 8)
        y_scale : str, optional
            [description], by default "log"
        which_features : str, optional
            [description], by default "all"
        display : bool, optional
            [description], by default True
        """
        # data from wide to long
        data = self.history_x.iloc[1:]
        data["index"] = data.index
        data = pd.melt(data, id_vars="index", var_name="Methods")

        decision_mapper = self.create_mapping_of_features_to_attribute(
            maps=["Tentative", "Rejected", "Accepted", "Shadow"]
        )
        data["Decision"] = data["Methods"].map(decision_mapper)
        data.drop(["index"], axis=1, inplace=True)

        options = {
            "accepted": self.filter_data(data, "Decision", "Accepted"),
            "tentative": self.filter_data(data, "Decision", "Tentative"),
            "rejected": self.filter_data(data, "Decision", "Rejected"),
            "all": data,
        }

        self.check_if_which_features_is_correct(which_features)
        data = options[which_features.lower()]

        self.box_plot(
            data=data,
            x_rotation=x_rotation,
            x_size=x_size,
            y_scale=y_scale,
            figsize=figsize,
        )
        if display:
            plt.show()
        else:
            plt.close()

    def box_plot(self, data, x_rotation, x_size, y_scale, figsize):
        """
        [summary]

        Parameters
        ----------
        data : [type]
            [description]
        x_rotation : [type]
            [description]
        x_size : [type]
            [description]
        y_scale : [type]
            [description]
        figsize : [type]
            [description]
        """
        if y_scale == "log":
            minimum = data["value"].min()
            if minimum <= 0:
                data["value"] += abs(minimum) + 0.01
        order = (
            data.groupby(by=["Methods"])["value"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        my_palette = self.create_mapping_of_features_to_attribute(
            maps=["yellow", "red", "green", "blue"]
        )

        # Use a color palette
        plt.figure(figsize=figsize)
        ax = sns.boxplot(
            x=data["Methods"], y=data["value"], order=order, palette=my_palette
        )

        if y_scale == "log":
            ax.set(yscale="log")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation, size=x_size)
        ax.set_title("Feature Importance")
        ax.set_ylabel("Z-Score")
        ax.set_xlabel("Features")

    def create_mapping_of_features_to_attribute(self, maps=None):
        """
        [summary]

        Parameters
        ----------
        maps : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        rejected = list(self.rejected)
        tentative = list(self.tentative)
        accepted = list(self.accepted)
        shadow = ["max_shadow", "median_shadow", "min_shadow", "mean_shadow"]

        tentative_map = self.create_list(tentative, maps[0])
        rejected_map = self.create_list(rejected, maps[1])
        accepted_map = self.create_list(accepted, maps[2])
        shadow_map = self.create_list(shadow, maps[3])

        values = tentative_map + rejected_map + accepted_map + shadow_map
        keys = tentative + rejected + accepted + shadow

        return self.to_dictionary(keys, values)

    @staticmethod
    def to_dictionary(list_one, list_two):
        """
        [summary]

        Parameters
        ----------
        list_one : [type]
            [description]
        list_two : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        return dict(zip(list_one, list_two))


def load_data(data_type="classification"):
    """
    [summary]

    Parameters
    ----------
    data_type : str, optional
        [description], by default "classification"

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    data_type = data_type.lower()
    error_message = (
        "No data_type was specified, use either 'classification' or 'regression'"
    )
    assert data_type in ["classification", "regression"], error_message

    if data_type == "classification":
        cancer = load_breast_cancer()
        x = pd.DataFrame(
            np.c_[cancer["data"], cancer["target"]],
            columns=np.append(cancer["feature_names"], ["target"]),
        )
        y = x.pop("target")
    else:
        boston = load_boston()
        x = pd.DataFrame(
            np.c_[boston["data"], boston["target"]],
            columns=np.append(boston["feature_names"], ["target"]),
        )
        y = x.pop("target")
    return x, y