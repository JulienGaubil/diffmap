from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import Model, ModelOutput
from ..model.projection import compute_track_flow
from ..tracking import Tracks
from .color import apply_color_map
from .drawing import draw_lines, draw_points
from .layout import add_border, hcat, vcat
from .visualizer import Visualizer


def generate_random_mask(
    desired: int,
    total: int,
    device: torch.device,
) -> Bool[Tensor, " total"]:
    if desired >= total:
        mask = torch.ones(total, dtype=torch.bool, device=device)

    mask = torch.zeros(total, dtype=torch.bool, device=device)
    indices = np.random.choice(total, desired, replace=False)
    indices = torch.tensor(indices, dtype=torch.int64, device=device)
    mask[indices] = True
    return mask


@dataclass
class VisualizerTrackingCfg:
    name: Literal["tracking"]
    num_vis_frames: int
    num_vis_points: int


class VisualizerTracking(Visualizer[VisualizerTrackingCfg]):
    def visualize(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        model: Model,
        global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"]]:
        return {}

        assert tracks is not None
        device = batch.videos.device

        # For now, only support batch size 1 for visualization.
        b, f, p, _ = tracks.xy.shape
        assert b == 1

        # Subsample the frames and points randomly.
        frame_mask = generate_random_mask(self.cfg.num_vis_frames, f, device)
        point_mask = generate_random_mask(self.cfg.num_vis_points, p, device)

        xy_track = tracks.xy[:, frame_mask][:, :, point_mask]
        xy_target, valid = compute_track_flow(
            model_output.surfaces[:, frame_mask],
            model_output.extrinsics[:, frame_mask],
            model_output.intrinsics[:, frame_mask],
            tracks[:, frame_mask][:, :, point_mask],
        )

        colors = torch.linspace(0, 1, self.cfg.num_vis_points, device=device)
        colors = apply_color_map(colors, "turbo")
        frames = batch.videos[0, frame_mask]
        visualization = []
        for i_source in range(self.cfg.num_vis_frames):
            row = []
            for i_target in range(self.cfg.num_vis_frames):
                source = frames[i_source]
                target = frames[i_target]
                mask = valid[0, i_source, i_target]

                # Handle the case where there are no valid points.
                if not mask.any():
                    cell = add_border(
                        add_border(vcat(source, target)),
                        border=1,
                        color=0,
                    )
                    row.append(cell)
                    continue

                # Draw ground-truth tracks on the source image.
                source = draw_lines(
                    source,
                    xy_track[0, i_source][mask],
                    xy_track[0, i_target][mask],
                    colors[mask],
                    2,
                    "round",
                    x_range=(0, 1),
                    y_range=(0, 1),
                )
                source = draw_points(
                    source,
                    xy_track[0, i_target][mask],
                    colors[mask],
                    radius=2,
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                # Draw lines between the ground-truth and predicted tracks on the target
                # image.
                target = draw_lines(
                    target,
                    xy_target[0, i_source, i_target][mask],
                    xy_track[0, i_target][mask],
                    colors[mask],
                    2,
                    "round",
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                # Draw small points on the target image for predicted points. Draw
                # big outlined points for ground-truth points. If they match up, you get
                # completely filled-in points.
                points = (
                    xy_target[0, i_source, i_target][mask],
                    xy_track[0, i_target][mask],
                )
                points = torch.cat(points, dim=0)
                stacked_colors = torch.cat((colors[mask], colors[mask]), dim=0)
                radii = (
                    torch.full_like(colors[mask][:, 0], 2),
                    torch.full_like(colors[mask][:, 0], 3),
                )
                radii = torch.cat(radii)
                inner_radii = (
                    torch.full_like(colors[mask][:, 0], 0),
                    torch.full_like(colors[mask][:, 0], 2),
                )
                inner_radii = torch.cat(inner_radii)
                target = draw_points(
                    target,
                    points,
                    stacked_colors,
                    radius=radii,
                    inner_radius=inner_radii,
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                cell = add_border(add_border(vcat(source, target)), border=1, color=0)
                row.append(cell)
            visualization.append(hcat(*row))
        return {"tracking": add_border(vcat(*visualization))}
