from .acs import *
from .adult import *
from .brfss import *
from .communities_and_crime import *
from .compas import *
from .german import *

# Default features for each dataset. These should describe the schema of the
# dataset AFTER the preprocess_fn is applied.
_DEFAULT_FEATURES = {
    "acsincome": ACS_INCOME_FEATURES,
    "brfss": BRFSS_FEATURES,
    "adult": ADULT_FEATURES,
    "communities_and_crime": CANDC_FEATURES,
    "compas": COMPAS_FEATURES,
    "german": GERMAN_FEATURES,
}
