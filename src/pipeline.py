# Third-party library imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Local imports
from .custom_transformers import (
    MissingValueChecker, 
    MissingValueStandardizer, 
    RobustSimpleImputer,
    SnakeCaseFormatter, 
    BooleanColumnTransformer, 
    JobStabilityTransformer, 
    CityTierTransformer, 
    StateDefaultRateTargetEncoder,
    RobustStandardScaler,
    RobustOneHotEncoder,
    RobustOrdinalEncoder,
    FeatureSelector
)
from .global_constants import (
    CRITICAL_FEATURES, 
    NON_CRITICAL_FEATURES, 
    COLUMNS_FOR_SNAKE_CASING,
    BOOLEAN_COLUMN_MAPPINGS,
    JOB_STABILITY_MAP,
    CITY_TIER_MAP,
    NUMERICAL_COLUMNS, 
    NOMINAL_COLUMN_CATEGORIES, 
    ORDINAL_COLUMN_ORDERS, 
    COLUMNS_TO_KEEP,
    RF_BEST_PARAMS
)


# --- Helper Functions to Create Full Pipeline and Pipeline Segments ---
def create_data_preprocessing_and_model_pipeline(): 
    return Pipeline([
        ("missing_value_checker", MissingValueChecker(critical_features=CRITICAL_FEATURES, non_critical_features=NON_CRITICAL_FEATURES)),
        ("missing_value_standardizer", MissingValueStandardizer()),
        ("missing_value_handler", ColumnTransformer(
            transformers=[("categorical_imputer", RobustSimpleImputer(strategy="most_frequent").set_output(transform="pandas"), NON_CRITICAL_FEATURES)],
            remainder="passthrough",
            verbose_feature_names_out=False  # preserve input column names instead of adding prefix 
        ).set_output(transform="pandas")),  # output pd.DataFrame instead of np.array 
        ("snake_case_formatter", SnakeCaseFormatter(columns=COLUMNS_FOR_SNAKE_CASING)),
        ("boolean_column_transformer", BooleanColumnTransformer(boolean_column_mappings=BOOLEAN_COLUMN_MAPPINGS)),
        ("job_stability_transformer", JobStabilityTransformer(job_stability_map=JOB_STABILITY_MAP)),
        ("city_tier_transformer", CityTierTransformer(city_tier_map=CITY_TIER_MAP)),
        ("state_default_rate_target_encoder", StateDefaultRateTargetEncoder()),
        ("feature_scaler_encoder", ColumnTransformer(
            transformers=[
                ("scaler", RobustStandardScaler(), NUMERICAL_COLUMNS), 
                ("nominal_encoder", RobustOneHotEncoder(categories=NOMINAL_COLUMN_CATEGORIES, drop="first", sparse_output=False), ["house_ownership"]),
                ("ordinal_encoder", RobustOrdinalEncoder(categories=ORDINAL_COLUMN_ORDERS), ["job_stability", "city_tier"])  
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        ).set_output(transform="pandas")),
        ("feature_selector", FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP)),
        ("rf_classifier", RandomForestClassifier(**RF_BEST_PARAMS, random_state=42)) 
    ])

def create_data_preprocessing_pipeline(): 
    return Pipeline([
    ("missing_value_checker", MissingValueChecker(critical_features=CRITICAL_FEATURES, non_critical_features=NON_CRITICAL_FEATURES)),
    ("missing_value_standardizer", MissingValueStandardizer()),
    ("missing_value_handler", ColumnTransformer(
        transformers=[("categorical_imputer", RobustSimpleImputer(strategy="most_frequent").set_output(transform="pandas"), NON_CRITICAL_FEATURES)],
        remainder="passthrough",
        verbose_feature_names_out=False  # preserve input column names instead of adding prefix 
    ).set_output(transform="pandas")),  # output pd.DataFrame instead of np.array 
    ("snake_case_formatter", SnakeCaseFormatter(columns=COLUMNS_FOR_SNAKE_CASING)),
    ("boolean_column_transformer", BooleanColumnTransformer(boolean_column_mappings=BOOLEAN_COLUMN_MAPPINGS)),
    ("job_stability_transformer", JobStabilityTransformer(job_stability_map=JOB_STABILITY_MAP)),
    ("city_tier_transformer", CityTierTransformer(city_tier_map=CITY_TIER_MAP)),
    ("state_default_rate_target_encoder", StateDefaultRateTargetEncoder()),
    ("feature_scaler_encoder", ColumnTransformer(
        transformers=[
            ("scaler", RobustStandardScaler(), NUMERICAL_COLUMNS), 
            ("nominal_encoder", RobustOneHotEncoder(categories=NOMINAL_COLUMN_CATEGORIES, drop="first", sparse_output=False), ["house_ownership"]),
            ("ordinal_encoder", RobustOrdinalEncoder(categories=ORDINAL_COLUMN_ORDERS), ["job_stability", "city_tier"])  
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")),
    ("feature_selector", FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP))
])

def create_model_preprocessing_pipeline(): 
    return Pipeline([
        ("feature_scaler_encoder", ColumnTransformer(
            transformers=[
                ("scaler", RobustStandardScaler(), NUMERICAL_COLUMNS), 
                ("nominal_encoder", RobustOneHotEncoder(categories=NOMINAL_COLUMN_CATEGORIES, drop="first", sparse_output=False), ["house_ownership"]),
                ("ordinal_encoder", RobustOrdinalEncoder(categories=ORDINAL_COLUMN_ORDERS), ["job_stability", "city_tier"])  
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        ).set_output(transform="pandas")),
        ("feature_selector", FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP))
    ])

def create_feature_engineering_pipeline(): 
    return Pipeline([
        ("snake_case_formatter", SnakeCaseFormatter(columns=COLUMNS_FOR_SNAKE_CASING)),
        ("boolean_column_transformer", BooleanColumnTransformer(boolean_column_mappings=BOOLEAN_COLUMN_MAPPINGS)),
        ("job_stability_transformer", JobStabilityTransformer(job_stability_map=JOB_STABILITY_MAP)),
        ("city_tier_transformer", CityTierTransformer(city_tier_map=CITY_TIER_MAP)),
        ("state_default_rate_target_encoder", StateDefaultRateTargetEncoder()),
    ])

def create_missing_value_handling_pipeline(): 
    return Pipeline([
        ("missing_value_checker", MissingValueChecker(critical_features=CRITICAL_FEATURES, non_critical_features=NON_CRITICAL_FEATURES)),
        ("missing_value_standardizer", MissingValueStandardizer()),
        ("missing_value_handler", ColumnTransformer(
            transformers=[("categorical_imputer", RobustSimpleImputer(strategy="most_frequent").set_output(transform="pandas"), NON_CRITICAL_FEATURES)],
            remainder="passthrough",
            verbose_feature_names_out=False  # preserve input column names instead of adding prefix 
        ).set_output(transform="pandas")),  # output pd.DataFrame instead of np.array 
    ])
