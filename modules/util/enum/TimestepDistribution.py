from enum import Enum


class TimestepDistribution(Enum):
    UNIFORM = 'UNIFORM'
    SIGMOID = 'SIGMOID'
    LOGIT_NORMAL = 'LOGIT_NORMAL'
    HEAVY_TAIL = 'HEAVY_TAIL'
    COS_MAP = 'COS_MAP'
    INVERTED_PARABOLA = 'INVERTED_PARABOLA'
    NEG_SQUARE = 'NEG_SQUARE'  # -x² distribution for balanced high/low noise training
    BETA = 'BETA'

    def __str__(self):
        return self.value
