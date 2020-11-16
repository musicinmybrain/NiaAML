from niaaml import data
from niaaml import classifiers
from niaaml import preprocessing_algorithms
from niaaml import feature_selection_algorithms
from niaaml.utilities import get_label_encoder
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from niaaml.utilities import Factory
from niaaml.task import Task
from niaaml.pipeline_optimizer import PipelineOptimizer
from niaaml.pipeline import Pipeline

__all__ = [
    'data',
    'preprocessing_algorithms',
    'feature_selection_algorithms',
    'get_label_encoder',
    'float_converter',
    'MinMax',
    'ParameterDefinition',
    'Factory',
    'PipelineOptimizer',
    'Pipeline'
]
__project__ = 'niaaml'
__version__ = '0.1.0'
