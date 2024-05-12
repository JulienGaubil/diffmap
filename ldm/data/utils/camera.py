"""Pinhole camera transformation and projection operations implemented using PyTorch."""
import torch
import numpy as np
from torch import Tensor
from jaxtyping import Float, Int
from dataclasses import dataclass

# from utils.util import to_euclidean_space, to_projective_space

def to_euclidean_space(homogeneous_points: Float[Tensor, "N+1 point"]) -> Float[Tensor, "N point"]:
    """Gets N-D Euclidean coordinates for points in associated projective space."""
    # Get homogeneous point with scale 1.
    homogeneous_points = homogeneous_points / homogeneous_points[-1,:].view(1,-1)

    # Return Euclidean coordinates.
    points = homogeneous_points[:-1,:]
    return points

def to_projective_space(points: Float[Tensor, "N point"]) -> Float[Tensor, "N+1 point"]:
    """Gets homogeneous coordinates in projective space with arbitrarily fixed scale = 1 on last coordinate
    for points from an N-dimensional Euclidean space.
    """
    # Scale in homogeneous coordinates chosen arbitrarily, scene coordinates are up to scale.
    scale =  torch.ones(1, points.size(1))
    return torch.cat([points, scale], dim=0)

def pixel_grid_coordinates(H: int, W: int) -> Int[Tensor, "height=H width=W uv=2"]:
    """For an image of with resolution H, W, create a grid ox pixel coordinates ranging from [0,H] and [0,W] with origin (0,0) in top left corner.
    x-axis is pointing right, y-axis is pointing down, bottom right corner at (H,W).
    uv_coordinates[:,:,0] corresponds to pixel height indices (y in image plane space), uv_coordinates[:,:,1] corresponds to pixel width indices (x in image plane space)
    """
    u, v = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing='ij'
    )

    uv_coordinates = torch.stack([u,v], dim=-1)
    return uv_coordinates

@dataclass
class Intrinsics:
    """Define camera intrinsics.
    """
    cx: float
    cy: float
    fx: float
    fy: float
    resolution: list[int,int] #(H,W)
    skew: float = 0

@dataclass
class Extrinsics:
    """Define an SE(3) transformation, convention world-to-camera transformation for extrinsics.
    """
    R: Float[Tensor, "3 3"]
    t: Float[Tensor, "3"]

class Camera:
    """A class that implement camera transformations defined from extrinsics with convention world-to-camera transformation and with OpenCV coordinate system for camera.
    """
    def __init__(
            self,
            intrinsics: Intrinsics,
            extrinsics: Extrinsics
        ) -> None:
        self.intrinsics = intrinsics
        self.extrinsincs = extrinsics

    @property
    def K(self) -> Float[Tensor, "3 3"]:
        """Intrisic matrix.
        """
        K = torch.eye(3, dtype=self.dtype)
        K[0,0] = self.intrinsics.fx
        K[1,1] = self.intrinsics.fy
        K[0,2] = self.intrinsics.cx
        K[1,2] = self.intrinsics.cy
        K[0,1] = self.intrinsics.skew

        return K
    
    @property
    def K_inv(self) -> Float[Tensor, "3 3"]:
        """Analytic inverse intrisic matrix.
        """
        K_inv = torch.eye(3, dtype=self.dtype)
        K_inv[0,0] = 1 / self.intrinsics.fx
        K_inv[1,1] = 1 / self.intrinsics.fy
        K_inv[0,1] = - self.intrinsics.skew / (self.intrinsics.fx * self.intrinsics.fy)
        K_inv[0,2] = (1 / self.intrinsics.fx) * (self.intrinsics.skew * self.intrinsics.cy / self.intrinsics.fy - self.intrinsics.cx)
        K_inv[1,2] = - self.intrinsics.cy / self.intrinsics.fy

        return K_inv
    
    @property
    def t(self) -> Float[Tensor, "3 1"]:
        return self.extrinsincs.t.view(3,1)
    
    @property
    def R(self) -> Float[Tensor, "3 3"]:
        return self.extrinsincs.R
    
    @property
    def w2c(self) ->  Float[Tensor, "4 4"]:
        """World-to-camera coordinates transformation matrix in the projective spaces associated to the 3D-Euclidean spaces.
        """
        P = torch.eye(4, dtype=self.dtype)
        P[:3,:3] = self.R
        P[:3,3] = self.t.squeeze()
        return P
    
    @property
    def c2w(self) ->  Float[Tensor, "4 4"]:
        """Camera-to-world coordinates transformation matrix in the projective spaces associated to the 3D-Euclidean spaces.
        """
        return torch.linalg.inv(self.w2c)
    
    @property
    def dtype(self) ->  torch.dtype:
        return self.R.dtype
    
    # @property
    # def coordinate_system(self) -> str:
    #     """Camera coordinate system."""
    #     return 'opencv' if self.R[1,1] < 0 else 'opengl'  ==> probably wrong?
    
    #################### Un-projection from 2D grid to world coordinates ###############################
    
    def pixel_to_im_plane(
            self,
            pixel_coordinates: Int[Tensor, "uv=2 pixel"]
    ) -> Float[Tensor, "xyw=3 point"]:
        """Convert pixel grid coordinates to points of the projective space associated to the 2D-Euclidean image plane space. pixel_coordinates[:,0] = coordinate along pixel grid height axis,
        pixel_coordinates[:,1] = coordinate along pixel grid width axis. Implemented for OpenCV camera coordinate system.
        """
        # Convert pixel to image plane coordinates in .
        im_plane_coordinates = pixel_coordinates[[1,0],:]
        
        # Get image plane homogeneous point coordinates.
        im_plane_pts = to_projective_space(im_plane_coordinates) #(3,N)
        return im_plane_pts.to(self.dtype)

    def im_plane_to_camera(
            self,
            im_plane_pts: Float[Tensor, "xyw=3 point"],
            depths: Float[Tensor, " 1 point"] | None = None
    ) -> Float[Tensor, "xyzw=4 point"]:
        """Unproject points from the projective space associated to the 2D-Euclidean image plane space to the projective space associated
        to the 3D-Euclidean local camera space.
        If depth is specified, projects to the associated point, else its **3D-Euclidean** local coordinates are defined up to scale.
        """
        # Enforce scale = 1 in image plane projective space and applies plane SE(2) transform.
        im_plane_pts = im_plane_pts / im_plane_pts[2,:].view(1,-1)
        homogeneous_plane_pts = self.K_inv @ im_plane_pts

        # Unproject to 3D Euclidean space - if depth not specified, assume depth = 1 and **3D-Euclidean** local coordinates are then defined up to scale.
        if depths is None:
            depths = torch.ones(1, homogeneous_plane_pts.size(1), dtype=self.dtype)
        local_coordinates = homogeneous_plane_pts * depths
        
        # Get homogeneous local camera coordinates.
        local_pts = to_projective_space(local_coordinates)

        return local_pts
    
    def camera_to_world(
            self,
            local_pts:  Float[Tensor, "xyzw=4 point"]
    ) -> Float[Tensor, "xyzw=4 point"]:
        """Transform points from projective space associated to the 3D-Euclidean local camera space to the projective space associated to the
        3D-Euclidean world space.
        """
        # Enforce scale = 1 in local camera projective space.
        local_pts = local_pts / local_pts[3,:].view(1,-1)

        # Cam to world transformation.
        world_pts = self.c2w @ local_pts
        return world_pts

    def pixel_to_world(
            self,
            pixel_coordinates: Int[Tensor, "uv=2 pixel"] | Float[Tensor, "uv=2 pixel"],
            depths: Float[Tensor, " 1 point=pixel"] | None = None
    ) -> Float[Tensor, "xyzw=4 point"]:
        """Unproject pixels in grid coordinates to the projective space associated to the 3D-Euclidean world space.
        """
        im_plane_pts = self.pixel_to_im_plane(pixel_coordinates)
        local_pts = self.im_plane_to_camera(im_plane_pts, depths)
        world_pts =  self.camera_to_world(local_pts)
        return world_pts
    
    #################### Projection from world to 2D pixel grid coordinates ###############################
    
    def world_to_local(
            self,
            world_pts: Float[Tensor, "xyzw=4 point"]
    ) -> Float[Tensor, "xyzw=4 point"]:
        """Transform points from the projective space associated to the 3D-Euclidean world space to the projective space associated to the
        3D-Euclidean local camera space.
        """
        # Enforce scale = 1 in world projective space.
        world_pts = world_pts / world_pts[3,:].view(1,-1)

        # World to cam transformation.
        local_pts = self.w2c @ world_pts
        return local_pts

    def local_to_im_plane(
            self,
            local_pts: Float[Tensor, "xyzw=4 point"]
    ) -> tuple[Float[Tensor, "xyw=3 point"], Float[Tensor, "1 point=pixel"]]:
        """Project points from the projective space associated to the projective space associated to the 3D-Euclidean local camera space 
        to points in the projective space associated to the image plane space with their depth in local camera coordinates.
        """
        # Enforce scale = 1 in local camera projective space.
        local_pts = local_pts / local_pts[3,:].view(1,-1) #ensures homogeneous points 

        # Projection on the projective space associated to a plane with projective scale = 1.
        local_depths = local_pts[2,:].view(1,-1)
        projected = local_pts[:-1,:]
        plane_pts = projected / projected[-1,:].view(1,-1)

        # Transformation to the image projective plane.
        im_plane_pts = self.K @ plane_pts

        return im_plane_pts, local_depths

    def world_to_im_plane(
            self,
            world_pts: Float[Tensor, "xyzw=4 point"]
    ) -> tuple[Float[Tensor, "xyw=3 point"], Float[Tensor, "1 point"]]:
        """Project points from the projective space associated to the 3D-Euclidean world space to 2D-Euclidean image plane coordinates with their depth
        in local camera coordinates.
        """
        local_pts = self.world_to_local(world_pts)
        im_plane_pts, local_depths = self.local_to_im_plane(local_pts)
        return im_plane_pts, local_depths
    
    def im_plane_to_pixel_coordinates(
            self,
            im_plane_pts: Float[Tensor, "xyw=3 point"],
            round_coords: bool = True
    ) -> Int[Tensor, "uv=2 pixel"] | Float[Tensor, "uv=2 pixel"]:
        """Convert points of the projective space associated to the 2D-Euclidean image plane space to pixel grid coordinates. pixel_coordinates[:,0] = coordinate along height grid axis,
        pixel_coordinates[:,1] = coordinate along pixel grid width axis. Implemented for OpenCV camera coordinate system.
        """
        # Get 2D-Euclidean image plane coordinates.
        im_plane_coordinates = to_euclidean_space(im_plane_pts)

        # Get pixel grid coordinates.
        pixel_coordinates = im_plane_coordinates[[1,0],:]
        if round_coords:
            pixel_coordinates = torch.round(pixel_coordinates)
        return pixel_coordinates
    
    def world_to_pixel_coordinates(
            self,
            world_pts: Float[Tensor, "xyzw=4 point"],
            round_coords: bool = True
    ) -> tuple[Int[Tensor, "uv=2 pixel"], Float[Tensor, "1 point=pixel"]]:
        """"Project points from the projective space associated to the 3D-Euclidean world space to pixel grid coordinates with their depth
        in local camera coordinates.
        """
        im_plane_coordinates, local_depths = self.world_to_im_plane(world_pts)
        pixels_coordinates = self.im_plane_to_pixel_coordinates(im_plane_coordinates, round_coords)

        return pixels_coordinates, local_depths
    
    def convert(self):
        "Inverts y,z axis in coordinate system, conversion OpenGL <--> OpenCV coordinate system"
        transf = torch.diag([1,-1,-1], dtype=self.dtype)
        self.extrinsincs.R = transf @ self.R
        self.extrinsincs.t = (transf @ self.t).resize(3)