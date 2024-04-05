from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from jaxtyping import Float
from matplotlib.figure import Figure
from torch import Tensor

from ..dataset.types import Batch
from ..flow import Flows
from ..misc.ate import compute_ate
from ..misc.image_io import fig_to_image
from ..model.model import Model, ModelOutput
from ..tracking import Tracks
from .layout import add_border
from .visualizer import Visualizer


def generate_plot(
    trajectories: list[Float[np.ndarray, "frame 3"]],
    labels: list[str],
    margin: float = 0.2,
) -> Figure:
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_proj_type("ortho")
    for trajectory, label in zip(trajectories, labels):
        xyz = rearrange(trajectory, "f xyz -> xyz f")
        ax.plot3D(*xyz, label=label)

    # Set square axes.
    points = np.concatenate(trajectories)
    minima = points.min(axis=0)
    maxima = points.max(axis=0)
    span = (maxima - minima).max() * (1 + margin)
    means = 0.5 * (maxima + minima)
    starts = means - 0.5 * span
    ends = means + 0.5 * span
    ax.set_xlim(starts[0], ends[0])
    ax.set_ylim(starts[1], ends[1])
    ax.set_zlim(starts[2], ends[2])
    fig.legend()
    return fig


@dataclass
class VisualizerTrajectoryCfg:
    name: Literal["trajectory"]


class VisualizerTrajectory(Visualizer[VisualizerTrajectoryCfg]):
    def visualize(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        model: Model,
        global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"]]:
        # If there's no ground truth, do nothing.
        if batch.extrinsics is None:
            return {}

        # For now, only support batch size 1 for visualization.
        b, _, _, _, _ = batch.videos.shape
        assert b == 1

        # Compute the ATE.
        try:
            ate, positions_gt, positions_hat = compute_ate(
                batch.extrinsics[0, :, :3, 3],
                model_output.extrinsics[0, :, :3, 3],
            )
        except ValueError:
            return {}

        # Visualize the trajectory.
        fg = generate_plot(
            [positions_gt.cpu().numpy(), positions_hat.cpu().numpy()],
            ["Ground-truth", "Predicted"],
        )
        visualization = fig_to_image(fg)
        plt.close(fg)

        return {
            "trajectory": add_border(visualization),
            "metrics/ate": ate,
        }
