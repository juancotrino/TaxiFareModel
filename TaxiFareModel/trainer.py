# imports
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_validate
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse

import mlflow
from  mlflow.tracking import MlflowClient
from memoized_property import memoized_property

import joblib

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[DE] [Munich] [juancotrino] TaxiFare V0.1"

class Trainer():

    def __init__(self, X, y, estimator):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME
        self.estimator = estimator

    def select_estimator(self):
        if self.estimator == 'LinearRegression':
            return LinearRegression()
        elif self.estimator == 'KNeighborsRegressor':
            return KNeighborsRegressor()
        elif self.estimator == 'Lasso':
            return Lasso()
        elif self.estimator == 'SGDRegressor':
            return SGDRegressor()


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            (self.estimator, self.select_estimator())
        ])
        return pipe

    def run(self):
        """set and train the pipeline"""
        return self.set_pipeline().fit(self.X, self.y)

    def cross_val(self):
        """cross vaalidates the model"""
        return cross_validate(self.select_estimator(), self.X, self.y)['test_score'].mean()

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.run().predict(X_test)
        return compute_rmse(y_pred, y_test)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.run(), EXPERIMENT_NAME + '_' + estimator + '.joblib')

    # MLFlow integration
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df = clean_data(df)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # train
    estimators = ['LinearRegression', 'KNeighborsRegressor', 'Lasso', 'SGDRegressor']

    for estimator in estimators:
        trainer = Trainer(X_train, y_train, estimator)
        trainer.set_pipeline()
        trainer.run()

        # cross validate
        #cv_score = trainer.cross_val()

        # evaluate
        result = trainer.evaluate(X_val, y_val)

        # MLFlow
        trainer.mlflow_log_param("Student Name", 'Juan Cotrino')
        trainer.mlflow_log_param('Model', estimator)
        trainer.mlflow_log_metric('RMSE', result)
        #trainer.mlflow_log_metric('CV Score', cv_score)

        # save model
        trainer.save_model()

    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{experiment_id}")
