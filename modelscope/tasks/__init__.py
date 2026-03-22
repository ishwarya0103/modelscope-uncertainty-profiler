"""Task evaluators."""

from modelscope.tasks.classification import ClassificationEvaluator
from modelscope.tasks.image_to_image import ImageToImageEvaluator
from modelscope.tasks.regression import RegressionEvaluator

__all__ = ["ClassificationEvaluator", "ImageToImageEvaluator", "RegressionEvaluator"]
