import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from causalml.inference.meta import BaseXClassifier
from causalml.inference.tree import UpliftRandomForestClassifier, UpliftTreeClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot

from xgboost import XGBRegressor, XGBClassifier

pd.set_option('display.max_columns', None)

class CausalChurnModel:

    def __init__(self, data, id_columns, treatment_column, outcome_column, control_column, objective):

        self.data = data

        self.id_columns = id_columns
        self.objective = objective

        self.treatment_column = treatment_column
        self.outcome_column = outcome_column

        self.control_column = control_column

        self.treatments_ = None
        self.categorical_column_names = None
        self.encoder = None
        self.uplift_model = None

    def validate_model_(self, f_test, t_test, o_test):

        predicted_uplift = self.uplift_model.predict(f_test)
        treatments = self.uplift_model.t_groups
        predicted_uplift = pd.DataFrame(predicted_uplift, columns=treatments)
        predicted_uplift[self.control_column] = 0

        if self.objective == "minimize":
            predicted_uplift["predicted_treatment"] = predicted_uplift.idxmin(axis=1)
        else:
            predicted_uplift["predicted_treatment"] = predicted_uplift.idxmax(axis=1)

        predicted_uplift["actual_treatment"] = t_test
        predicted_uplift["actual_outcome"] = o_test

        matching_case = predicted_uplift[predicted_uplift["predicted_treatment"]==
            predicted_uplift["actual_treatment"]]
        
        non_matching_case = predicted_uplift[predicted_uplift["predicted_treatment"]!=
            predicted_uplift["actual_treatment"]]
        
        matching_churn_rate = matching_case["actual_outcome"].mean()
        non_matching_churn_rate = non_matching_case["actual_outcome"].mean()
        average_churn_rate = predicted_uplift["actual_outcome"].mean()

        return {"Average Outcome": average_churn_rate, "Average Matching Outcome": matching_churn_rate
            ,"Average Non Matching Outcome": non_matching_churn_rate}
        
    def encode_data_(self, data=None, prediction=False):

        if data is not None:
            data = data

        else:
            data = self.data

        columns_ = set(data.columns)
        
        columns_to_encode_ = columns_ - set([self.treatment_column, self.outcome_column])
        columns_to_encode_ = list(columns_to_encode_ - set(self.id_columns))

        data_to_encode_ = data[columns_to_encode_]

        X_num = data_to_encode_.select_dtypes(exclude='object')
        X_cat = data_to_encode_.select_dtypes(include='object')

        if prediction:

            X_cat = pd.DataFrame(self.encoder.transform(X_cat)
            , columns=self.categorical_column_names)

            encoded_data = X_num.join(X_cat)
            encoded_data.fillna(0, inplace=True)
            encoded_data[self.treatment_column] = data[self.treatment_column]
            encoded_data[self.outcome_column] = data[self.outcome_column]

            for col_name in self.id_columns:
                encoded_data[col_name] = data[col_name]

            return encoded_data

        self.colums_to_encode = X_cat.columns

        enc = OneHotEncoder(sparse_output=False)
        X_cat = pd.DataFrame(enc.fit_transform(X_cat))

        self.encoder = enc

        categorical_columns = self.colums_to_encode

        X_cat = pd.DataFrame(X_cat, columns=enc.get_feature_names_out(categorical_columns))

        self.categorical_column_names = enc.get_feature_names_out(categorical_columns)

        encoded_data = X_num.join(X_cat)
        encoded_data.fillna(0, inplace=True)

        encoded_data[self.treatment_column] = data[self.treatment_column]
        encoded_data[self.outcome_column] = data[self.outcome_column]

        for col_name in self.id_columns:
            encoded_data[col_name] = data[col_name]

        return encoded_data

    def preprocess_data(self, data):

        encoded_data = self.encode_data_(data, prediction=True)
        ids = encoded_data[self.id_columns]
        features = encoded_data[self.column_structure]

        return ids, features

    def train_uplift_model(self):

        self.treatments_ = list(self.data[self.treatment_column].unique())

        encoded_data = self.encode_data_()

        to_omit = self.id_columns + [self.treatment_column, self.outcome_column]

        features = encoded_data.drop(to_omit, axis=1)
        treatments = encoded_data[self.treatment_column]
        outcome = encoded_data[self.outcome_column]

        f_train, f_test, t_train, t_test, o_train, o_test = train_test_split(features
        , treatments, outcome, test_size=0.2)

        self.column_structure = list(f_train.columns)

        ul_model = BaseXClassifier(
            control_outcome_learner=XGBClassifier(),
            treatment_outcome_learner=XGBClassifier(),
            control_effect_learner=XGBRegressor(),
            treatment_effect_learner=XGBRegressor(),
            ate_alpha=0.05,
            control_name=self.control_column)
        
        ul_model.fit(X=f_train, treatment=t_train, y=o_train)

        self.uplift_model = ul_model

        validation_response = self.validate_model_(f_test, t_test, o_test)

        return { "success" : True, "validation_response" : validation_response }
    
    def find_best_treatment(self, data):

        data = data.reset_index(drop=True)

        ids, f_processed = self.preprocess_data(data)

        treatments = list(self.uplift_model.t_groups)
        predicted_uplift = self.uplift_model.predict(f_processed)

        predicted_uplift = pd.DataFrame(predicted_uplift, columns=treatments)
        predicted_uplift[self.control_column] = 0

        predicted_uplift_out = ids.join(predicted_uplift)

        if self.objective == "minimize":
            predicted_uplift_out["best_predicted_treatment"] = predicted_uplift.idxmin(axis=1)
        
        else:
            predicted_uplift_out["best_predicted_treatment"] = predicted_uplift.idxmax(axis=1)

        return predicted_uplift_out

    def find_uplift_of_a_segment_(self, data, segment):

        data = data.reset_index(drop=True)

        ids, f_processed = self.preprocess_data(data)

        segments = set(data[segment].unique())

        treatments = list(self.uplift_model.t_groups)
        predicted_uplift = self.uplift_model.predict(f_processed)

        predicted_uplift = pd.DataFrame(predicted_uplift, columns=treatments)
        predicted_uplift["segment"] = data[segment]

        p_uplit = predicted_uplift.groupby("segment").mean()

        return p_uplit
    