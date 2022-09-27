from .adult import *
from .compas import *
from .german import *

# Default features for each dataset
_DEFAULT_FEATURES = {
    "adult": ADULT_FEATURES,
    "compas": COMPAS_FEATURES,
    "german": GERMAN_FEATURES,
}
