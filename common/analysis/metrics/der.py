from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Segment, Annotation

import warnings
# Here, we want to filter the following warning :
# UserWarning: 'uem' was approximated by the union of 'reference' and 'hypothesis' extents.
# "'uem' was approximated by the union of 'reference'"
warnings.filterwarnings("ignore", category=UserWarning, module='pyannote')

def diarization_error_rate(y_true, y_pred, times):
    """
    :param y_true: Ground truth speakers per utterance
    :param y_pred: Predicted speakers per utterance
    :param times: time per utterance in seconds (the seconds can be a float, they will be converted and rounded to
        integer milliseconds)
    :return: The Diarization Error Rate (DER)
    """
    metric = DiarizationErrorRate()
    reference = _generate_annotations(y_true, times)
    hypothesis = _generate_annotations(y_pred, times)
    value = metric(reference, hypothesis)
    return value

def _generate_annotations(utterances, times):
    annotation = Annotation()
    start = 0
    for u, utterance in enumerate(utterances):
        end = start + round(times[u] * 1000)
        annotation[Segment(start, end)] = utterance
        start = end
    return annotation
