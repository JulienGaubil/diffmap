from .visualizer import Visualizer
from .visualizer_summary import VisualizerSummary, VisualizerSummaryCfg
from .visualizer_tracking import VisualizerTracking, VisualizerTrackingCfg
from .visualizer_trajectory import VisualizerTrajectory, VisualizerTrajectoryCfg

VISUALIZERS = {
    "summary": VisualizerSummary,
    "trajectory": VisualizerTrajectory,
    "tracking": VisualizerTracking,
}

VisualizerCfg = VisualizerSummaryCfg | VisualizerTrajectoryCfg | VisualizerTrackingCfg


def get_visualizers(cfgs: list[VisualizerCfg]) -> list[Visualizer]:
    return [VISUALIZERS[cfg.name](cfg) for cfg in cfgs]
