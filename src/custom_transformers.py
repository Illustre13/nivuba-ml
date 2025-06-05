# Imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np


# --- Custom error classes --- 
# For missing values in critical columns of the X input DataFrame (in MissingValueChecker)
class MissingValueError(ValueError):
    pass

# For mistmatch between expected and actual columns in X input DataFrame because of missing columns, unexpected columns, or wrong column order 
class ColumnMismatchError(ValueError):
    pass

# For invalid categorical labels (in BooleanColumnTransformer)
class CategoricalLabelError(ValueError):
    pass


# --- Custom transformer classes for data preprocessing pipeline ---
# Check missing values 
class MissingValueChecker(BaseEstimator, TransformerMixin):
    def __init__(self, critical_features, non_critical_features):
        # Validate input data type
        if not isinstance(critical_features, list):
            raise TypeError("'critical_features' must be a list of column names.")
        if not isinstance(non_critical_features, list):
            raise TypeError("'non_critical_features' must be a list of column names.")

        # Validate input value
        if not critical_features:
            raise ValueError("'critical_features' cannot be an empty list. It must specify the names of the critical features.")
        if not non_critical_features:
            raise ValueError("'non_critical_features' cannot be an empty list. It must specify the names of the non-critical features.")

        self.critical_features = critical_features
        self.non_critical_features = non_critical_features
    
    def _validate_input(self, X):
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")          
        
        # Ensure input DataFrame contains all required columns
        input_columns = set(X.columns)
        required_columns = set(self.critical_features + self.non_critical_features)
        missing_columns = required_columns - input_columns 
        if missing_columns:
            raise ColumnMismatchError(f"Input X is missing the following columns: {', '.join(missing_columns)}.")

        # Ensure input DataFrame doesn't contain any unexpected columns
        unexpected_columns = input_columns - required_columns
        if unexpected_columns:
            raise ColumnMismatchError(f"Input X contains the following columns that are neither defined in 'critical_features' nor 'non_critical_features: {', '.join(unexpected_columns)}.")

    def _check_missing_values(self, X):
        # --- Critical features ---
        # Calculate total number of missing values  
        n_missing_total_critical = X[self.critical_features].isnull().sum().sum()
        # Calculate number of rows with missing values  
        n_missing_rows_critical = X[self.critical_features].isnull().any(axis=1).sum()
        # Create dictionary with number of missing values by column 
        n_missing_by_column_critical = X[self.critical_features].isnull().sum().to_dict()
        # Raise error  
        if n_missing_total_critical > 0:
            values = "value" if n_missing_total_critical == 1 else "values"
            rows = "row" if n_missing_rows_critical == 1 else "rows"
            raise MissingValueError(
                f"{n_missing_total_critical} missing {values} found in critical features "
                f"across {n_missing_rows_critical} {rows}. Please provide missing {values}.\n"
                f"Missing values by column: {n_missing_by_column_critical}"
            )

        # --- Non-critical features ---
        # Calculate total number of missing values 
        n_missing_total_noncritical = X[self.non_critical_features].isnull().sum().sum()        
        # Calculate number of rows with missing values 
        n_missing_rows_noncritical = X[self.non_critical_features].isnull().any(axis=1).sum()
        # Create dictionary with number of missing values by column 
        n_missing_by_column_noncritical = X[self.non_critical_features].isnull().sum().to_dict()
        # Display warning message
        if n_missing_total_noncritical > 0:
            values = "value" if n_missing_total_noncritical == 1 else "values"
            rows = "row" if n_missing_rows_noncritical == 1 else "rows"
            print(
                f"Warning: {n_missing_total_noncritical} missing {values} found in non-critical features "
                f"across {n_missing_rows_noncritical} {rows}. Missing {values} will be imputed.\n"
                f"Missing values by column: {n_missing_by_column_noncritical}"
            )
            
    def fit(self, X, y=None):
        # Validate input 
        self._validate_input(X)  

        # Check missing values
        self._check_missing_values(X)

        # Raise MissingValueError if a non-critical feature has only missing values
        for non_critical_feature in self.non_critical_features:
            if X[non_critical_feature].isnull().all():
                raise MissingValueError(f"'{non_critical_feature}' cannot be only missing values. Please ensure at least one non-missing value.")

        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        return self 

    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input 
        self._validate_input(X)    
        
        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ColumnMismatchError("Feature names and feature order of input X must be the same as during .fit().")      
        
        # Check missing values
        self._check_missing_values(X)

        return X


# Standardize missing values
class MissingValueStandardizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
               
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        return self
    
    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
            
        # Convert all missing value types (None, np.nan, pd.NA, etc.) to np.nan
        return X.fillna(value=np.nan)


# A wrapper for SimpleImputer to passthrough empty DataFrames during .transform() instead of raising a ValueError (SimpleImputer default behavior)
class RobustSimpleImputer(SimpleImputer):
    def transform(self, X):
        if X.empty:
            return X
        else:
            return super().transform(X)


# Format categorical labels in snake_case
class SnakeCaseFormatter(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        if not isinstance(columns, list) and columns is not None:
            raise TypeError("'columns' must be a list of column names or None. If None, all columns will be used.")

        # Validate input value
        if columns == []:
            raise ValueError("'columns' cannot be an empty list. It must specify the column names for snake case formatting.")

        self.columns = columns
    
    def fit(self, X, y=None):
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   

        # Determine columns to be transformed (all if none provided)
        if self.columns is None:
            self.columns_ = X.columns.tolist()
        else:
            self.columns_ = self.columns
            # Ensure input DataFrame contains all required columns
            missing_columns = set(self.columns_) - set(X.columns)
            if missing_columns:
                raise ColumnMismatchError(f"Input X is missing the following columns: {', '.join(missing_columns)}.")
        
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()
                    
        return self  

    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   

        # Ensure input DataFrame contains all required columns
        missing_columns = set(self.columns_) - set(X.columns)
        if missing_columns:
            raise ColumnMismatchError(f"Input X is missing the following columns: {', '.join(missing_columns)}.")

        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ColumnMismatchError("Feature names and feature order of input X must be the same as during .fit().")      

        X_transformed = X.copy()
        
        for column in self.columns_:
            X_transformed[column] = X_transformed[column].apply(
                lambda categorical_label: (
                    categorical_label
                    .strip()  # Remove leading/trailing spaces
                    .lower()  # Convert to lowercase
                    .replace("-", "_")  # Replace hyphens with "_"
                    .replace("/", "_")  # Replace slashes with "_"
                    .replace(" ", "_")  # Replace spaces with "_"
                    if isinstance(categorical_label, str) else categorical_label
                )
            )

        return X_transformed


# Convert binary categorical columns to boolean columns 
class BooleanColumnTransformer(BaseEstimator, TransformerMixin):  
    def __init__(self, boolean_column_mappings):
        # Validate input data type
        if not isinstance(boolean_column_mappings, dict):
            raise TypeError("'boolean_column_mappings' must be a dictionary specifying the mappings.")

        # Validate input value
        if not boolean_column_mappings:
            raise ValueError("'boolean_column_mappings' cannot be an empty dictionary. It must specify the the mappings.")

        # Iterate all columns in "boolean_column_mappings"
        for column, mapping in boolean_column_mappings.items():
            # Ensure the mapping of the current column is also a dictionary
            if not isinstance(mapping, dict):
                raise TypeError(f"The mapping for '{column}' must be a dictionary.")
            
            # Ensure the values of the current mapping are boolean
            if not all(isinstance(value, bool) for value in mapping.values()):
                raise ValueError(f"All values in the mapping for '{column}' must be boolean (True or False).")

        self.boolean_column_mappings = boolean_column_mappings 
            
    def _validate_input(self, X):
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
        
        # Ensure input DataFrame contains all required binary columns (from "boolean_column_mappings")
        input_columns = set(X.columns)
        required_columns = set(self.boolean_column_mappings.keys())
        missing_columns = required_columns - input_columns 
        if missing_columns:
            raise ColumnMismatchError(f"Input X is missing the following columns: {', '.join(missing_columns)}.")

        # Ensure all binary columns have no missing values
        for column in required_columns:
            if X[column].isna().any():
                raise MissingValueError(f"'{column}' column cannot contain missing values.")

        # Ensure all binary columns have valid data types (str, int, float, bool)
        for column in required_columns:
            if X[column].apply(lambda x: not isinstance(x, (str, int, float, bool))).any():
                raise TypeError(f"All values in '{column}' column must be str, int, float or bool.")

        # Ensure all binary columns contains only known labels (from "boolean_column_mappings")
        for column, mapping in self.boolean_column_mappings.items():            
            known_labels = set(mapping.keys())
            input_labels = set(X[column].unique())
            unknown_labels = input_labels- known_labels
            if unknown_labels:
                raise CategoricalLabelError(f"'{column}' column contains unknown labels that are not in 'boolean_column_mappings': {', '.join(unknown_labels)}.")

    def fit(self, X, y=None):  
        # Validate input
        self._validate_input(X)
        
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()
        
        return self  

    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input
        self._validate_input(X)
                    
        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ColumnMismatchError("Feature names and feature order of input X must be the same as during .fit().")      

        X_transformed = X.copy()
        for column, mapping in self.boolean_column_mappings.items():
            X_transformed[column] = X_transformed[column].map(mapping)

        return X_transformed


# Derive job stability from profession 
class JobStabilityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, job_stability_map):
        # Validate input data type
        if not isinstance(job_stability_map, dict):
            raise TypeError("'job_stability_map' must be a dictionary specifying the mappings from 'profession' to 'job_stability'.")
        
        # Validate input value
        if not job_stability_map:
            raise ValueError("'job_stability_map' cannot be an empty dictionary. It must specify the mappings from 'profession' to 'job_stability'.")

        self.job_stability_map = job_stability_map 

    def _validate_input(self, X):
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   

        # Ensure input DataFrame contains the required "profession" column
        if "profession" not in X.columns:
            raise ColumnMismatchError("Input X is missing the following columns: profession.")

        # Ensure "profession" column has no missing values
        if X["profession"].isna().any():
            raise MissingValueError("'profession' column cannot contain missing values.")

        # Ensure all values in "profession" column are strings
        if X["profession"].apply(lambda x: not isinstance(x, str)).any():
            raise TypeError("All values in 'profession' column must be strings.")

        # Ensure "profession" column contains only known professions (from "job_stability_map")
        known_professions = set(self.job_stability_map.keys())
        input_professions = set(X["profession"].unique())
        unknown_professions = input_professions - known_professions
        if unknown_professions:
            raise CategoricalLabelError(f"'profession' column contains unknown professions: {', '.join(unknown_professions)}.")

    def fit(self, X, y=None):
        # Validate input
        self._validate_input(X)

        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()
        
        return self  

    def transform(self, X):  
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input
        self._validate_input(X)
        
        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ColumnMismatchError("Feature names and feature order of input X must be the same as during .fit().")      
       
        # Create job stability column by mapping professions to job stability tiers (default to "moderate" for unknown professions)
        X_transformed = X.copy()
        X_transformed["job_stability"] = X_transformed["profession"].map(self.job_stability_map)

        return X_transformed


# Derive city tier from city
class CityTierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, city_tier_map):
        if not isinstance(city_tier_map, dict):
            raise TypeError("'city_tier_map' must be a dictionary specifying the mappings from 'city' to 'city_tier'.")

        # Validate input value
        if not city_tier_map:
            raise ValueError("'city_tier_map' cannot be an empty dictionary. It must specify the mappings from 'city' to 'city_tier'.")

        self.city_tier_map = city_tier_map 

    def _validate_input(self, X):
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")           

        # Ensure input DataFrame contains the required "city" column
        if "city" not in X.columns:
            raise ColumnMismatchError("Input X is missing the following columns: city.")
        
        # Ensure "city" column has no missing values
        if X["city"].isna().any():
            raise MissingValueError("'city' column cannot contain missing values.")

        # Ensure all values in "city" column are strings
        if X["city"].apply(lambda x: not isinstance(x, str)).any():
            raise TypeError("All values in 'city' column must be strings.")

        # Ensure "city" column contains only known cities (from "city_tier_map")
        known_cities = set(self.city_tier_map.keys())
        input_cities = set(X["city"].unique())
        unknown_cities = input_cities - known_cities
        if unknown_cities:
            raise CategoricalLabelError(f"'city' column contains unknown cities: {', '.join(unknown_cities)}.")
    
    def fit(self, X, y=None):
        # Validate input
        self._validate_input(X) 
            
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()
        
        return self  

    def transform(self, X):        
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input
        self._validate_input(X)   
        
        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ColumnMismatchError("Feature names and feature order of input X must be the same as during .fit().")      

        # Create city tier column by mapping cities to city tiers
        X_transformed = X.copy()
        X_transformed["city_tier"] = X_transformed["city"].map(self.city_tier_map)
        
        return X_transformed


# Target encoding of state default rate 
class StateDefaultRateTargetEncoder(BaseEstimator, TransformerMixin):
    def _validate_X_input(self, X):
        # Ensure X input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
        
        # Ensure DataFrame contains the required "state" column
        if "state" not in X.columns:
            raise ColumnMismatchError("Input X is missing the following columns: state.")
        
        # Ensure "state" column has no missing values
        if X["state"].isna().any():
            raise MissingValueError("'state' column cannot contain missing values.")
        
        # Ensure all values in "state" column are strings
        if X["state"].apply(lambda x: not isinstance(x, str)).any():
            raise TypeError("All values in 'state' column must be strings.")
                
    def fit(self, X, y):
        # Validate X input
        self._validate_X_input(X)

        # Ensure y input is a pandas Series
        if not isinstance(y, pd.Series):
            raise TypeError("Input y must be a pandas Series.")   
        
        # Ensure y has no missing values
        if y.isna().any():
            raise MissingValueError("Input y cannot contain missing values.")

        # Ensure y is integer type
        if not pd.api.types.is_integer_dtype(y):
            raise TypeError("Input y must be integer type.")

        # Ensure all y values are 0 or 1 
        if not y.isin([0, 1]).all():
            raise ValueError("All y values must be 0 (no default) or 1 (default).")
                
        # Ensure X and y have the same index
        if not X.index.equals(y.index):
            raise ValueError("Input X and y must have the same index.")

        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()
       
        # Calculate default rate by state
        df = X.copy()
        df["default"] = y
        self.default_rate_by_state_ = df.groupby("state")["default"].mean()
        
        return self  

    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)

        # Validate X input
        self._validate_X_input(X)

        # Ensure "state" column contains only known states seen during .fit()
        known_states = set(self.default_rate_by_state_.index)
        input_states = set(X["state"].unique())
        unknown_states = input_states - known_states
        if unknown_states:
            raise CategoricalLabelError(f"'state' column contains unknown states: {', '.join(unknown_states)}.")
      
        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ColumnMismatchError("Feature names and feature order of input X must be the same as during .fit().")      

        # Create state default rate column by mapping the state to its corresponding default rate
        X_transformed = X.copy()
        X_transformed["state_default_rate"] = X_transformed["state"].map(self.default_rate_by_state_)
        
        return X_transformed


# A wrapper for StandardScaler to passthrough empty DataFrames during .transform() instead of raising a ValueError
class RobustStandardScaler(StandardScaler):
    def transform(self, X):
        if X.empty:
            feature_names_out = self.get_feature_names_out(X.columns)
            return pd.DataFrame(columns=feature_names_out, dtype=float)
        else:
            return super().transform(X)


# A wrapper for StandardScaler to passthrough empty DataFrames during .transform() instead of raising a ValueError
class RobustOneHotEncoder(OneHotEncoder):
    def transform(self, X):
        check_is_fitted(self)
        if X.empty:
            feature_names_out = self.get_feature_names_out(X.columns)
            return pd.DataFrame(columns=feature_names_out, dtype=float)
        else:
            return super().transform(X)


# A wrapper for StandardScaler to passthrough empty DataFrames during .transform() instead of raising a ValueError
class RobustOrdinalEncoder(OrdinalEncoder):
    def transform(self, X):
        if X.empty:
            feature_names_out = self.get_feature_names_out(X.columns)
            return pd.DataFrame(columns=feature_names_out, dtype=float)
        else:
            return super().transform(X)


# Feature selection for downstream model training and inference 
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_keep):
        # Validate input data type
        if not isinstance(columns_to_keep, list):
            raise TypeError("'columns_to_keep' must be a list of column names.")

        # Validate input value
        if not columns_to_keep:
            raise ValueError("'columns_to_keep' cannot be an empty list. It must specify the column names.")

        self.columns_to_keep = columns_to_keep

    def fit(self, X, y=None):
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
        
        # Ensure input DataFrame contains all columns_to_keep
        missing_columns = set(self.columns_to_keep) - set(X.columns)
        if missing_columns:
            raise ColumnMismatchError(f"Input X is missing the following columns: {', '.join(missing_columns)}.")
            
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        return self  

    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)

        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ColumnMismatchError("Feature names and feature order of input X must be the same as during .fit().")      
        
        # Create transformed DataFrame with only the selected features
        X_transformed = X[self.columns_to_keep].copy()
        
        return X_transformed 