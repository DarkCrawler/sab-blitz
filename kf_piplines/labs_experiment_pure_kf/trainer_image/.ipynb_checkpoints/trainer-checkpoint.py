# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Covertype Classifier trainer script."""
import os
import pickle
import subprocess
import sys

import fire
import hypertune
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from google.cloud import bigquery

#AIP_MODEL_DIR = os.environ["AIP_MODEL_DIR"]
MODEL_FILENAME = "model.pkl"


def train_evaluate(
    alpha, max_iter, hptune
):
    bqclient = bigquery.Client()
    
    """Trains the Covertype Classifier model."""
    
    train_set_query = """
    SELECT
    *
    FROM `qwiklabs-gcp-00-59947c43422f.kfp_test.full_data` AS train_set WHERE MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(train_set))), 10) IN (1, 2, 3, 4)
    """
    
    eval_set_query = """
    SELECT
    *
    FROM `qwiklabs-gcp-00-59947c43422f.kfp_test.full_data` AS train_set WHERE MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(train_set))), 10) IN (2)
    """
    
    df_train=(
        bqclient.query(train_set_query)
        .result()
        .to_dataframe(
            # Optionally, explicitly request to use the BigQuery Storage API. As of
            # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
            # API is used by default.
            create_bqstorage_client=True,)
    )
    
    df_validation=(
        bqclient.query(eval_set_query)
        .result()
        .to_dataframe(
            # Optionally, explicitly request to use the BigQuery Storage API. As of
            # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
            # API is used by default.
            create_bqstorage_client=True,)
    )


    if not hptune:
        df_train = pd.concat([df_train, df_validation])

    numeric_feature_indexes = slice(0, 10)
    categorical_feature_indexes = slice(10, 12)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_feature_indexes),
            ("cat", OneHotEncoder(), categorical_feature_indexes),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", SGDClassifier(loss="log")),
        ]
    )

    num_features_type_map = {
        feature: "float64"
        for feature in df_train.columns[numeric_feature_indexes]
    }
    df_train = df_train.astype(num_features_type_map)
    df_validation = df_validation.astype(num_features_type_map)

    print(f"Starting training: alpha={alpha}, max_iter={max_iter}")
    # pylint: disable-next=invalid-name
    X_train = df_train.drop("Cover_Type", axis=1)
    y_train = df_train["Cover_Type"]

    pipeline.set_params(classifier__alpha=alpha, classifier__max_iter=max_iter)
    pipeline.fit(X_train, y_train)

    if hptune:
        # pylint: disable-next=invalid-name
        X_validation = df_validation.drop("Cover_Type", axis=1)
        y_validation = df_validation["Cover_Type"]
        accuracy = pipeline.score(X_validation, y_validation)
        print(f"Model accuracy: {accuracy}")
        # Log it with hypertune
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="accuracy", metric_value=accuracy
        )

    # Save the model
    # if not hptune:
    #     with open(MODEL_FILENAME, "wb") as model_file:
    #         pickle.dump(pipeline, model_file)
    #     subprocess.check_call(
    #         ["gsutil", "cp", MODEL_FILENAME, AIP_MODEL_DIR], stderr=sys.stdout
    #     )
    #     print(f"Saved model in: {AIP_MODEL_DIR}")


if __name__ == "__main__":
    fire.Fire(train_evaluate)
