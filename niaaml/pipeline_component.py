import numpy as np
from niaaml.utilities import MinMax

__all__ = [
	'PipelineComponent'
]

class PipelineComponent:
	r"""Class for implementing pipeline components.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	Attributes:
		_params (Dict[str, ParameterDefinition]): Dictionary of components's parameters with possible values. Possible parameter values are given as an instance of the ParameterDefinition class.
	
	See Also:
		* :class:`niaaml.utilities.ParameterDefinition`
    """

	_params = None

	def __init__(self, **kwargs):
		r"""Initialize pipeline component.
		"""
		self._set_parameters(**kwargs)
	
	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the pipeline component.
		"""
		return
	 
	@classmethod
	def getRandomInstance(i):
		r"""Randomly initialize instance of the implemented `niaaml.pipeline_component.PipelineComponent` class.

        Arguments:
            i (PipelineComponent): Any class that implements PipelineComponent class.

        Returns:
            PipelineComponent: Randomly initialized PipelineComponent instance.
		"""
		instance = i()
		params = dict()

		if i._params:
			for key, value in i._params.items():
				# value should be somehow determined runtime in case its value is currently None
				if value is not None:
					if isinstance(value.value, MinMax):
						val = np.random.uniform(value.value.min, value.value.max)
						if value.paramType is np.intc or value.paramType is np.uintc or value.paramType is np.uint:
							val = value.paramType(np.around(val))
						params[key] = val
					else:
						params[key] = value.value[np.random.randint(0, len(value.value))]
			
			instance._set_parameters(**params)
		
		return instance
