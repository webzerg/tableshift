from .adult import *

# Default features for each dataset
_DEFAULT_FEATURES = {
    "adult": ADULT_FEATURES,
}

_PREPROCESS_FNS = {
    "adult": preprocess_adult,
}

# List of resources needed for each UCI dataset.
_UCI_DATA_SOURCES = {
    "adult": ADULT_RESOURCES,
}