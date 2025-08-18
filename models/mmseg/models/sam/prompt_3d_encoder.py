import torch
from torch import nn
from typing import Optional, Tuple, Type

from .common import LayerNorm2d

class Prompt3dEncoder(nn.Module):
    def __init__(
        self,
        prompt_embed_dim: int,
        image_embedding_size: Tuple[int, int],
        voxel_size: int,
        # input_pointcloud_size: Tuple[int, int, int],  # Size of the point cloud grid (H, W, D)
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes 3D point clouds and bounding box proposals for input to a model.

        Arguments:
          embed_dim (int): The embedding dimension for points and masks.
          input_pointcloud_size (tuple(int, int, int)): The spatial size of the point cloud grid.
          mask_in_chans (int): The number of channels used for encoding input masks.
          activation (nn.Module): The activation to use when encoding input masks.
        """
        super().__init__()
        self.prompt_embed_dim = prompt_embed_dim
        self.input_pointcloud_size = image_embedding_size[0]//voxel_size, image_embedding_size[1]//voxel_size, 1

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=3, stride=1, padding=1),  # Keep spatial dimensions
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans // 2, kernel_size=3, stride=1, padding=1),  # Keep spatial dimensions
            LayerNorm2d(mask_in_chans // 2),
            activation(),
            nn.Conv2d(mask_in_chans // 2, mask_in_chans, kernel_size=3, stride=1, padding=1),  # Keep spatial dimensions
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, int(mask_in_chans*2), kernel_size=3, stride=1, padding=1),  # Keep spatial dimensions
            LayerNorm2d(int(mask_in_chans*2)),
            activation(),
            nn.Conv2d(int(mask_in_chans*2), int(mask_in_chans*4), kernel_size=3, stride=1, padding=1),  # Keep spatial dimensions
            LayerNorm2d(int(mask_in_chans*4)),
            activation(),
            nn.Conv2d(int(mask_in_chans*4), prompt_embed_dim, kernel_size=1),  # This doesn't affect spatial size
        )

        self.no_mask_embed = nn.Embedding(1, prompt_embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point clouds,
        applied to a dense set of points the shape of the point cloud encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(H)x(W)x(D)
        """
        return self.pe_layer(self.input_pointcloud_size).unsqueeze(0)

    def _extract_points_in_bbox(
        self, pointcloud: torch.Tensor, bbox: torch.Tensor
    ) -> torch.Tensor:
        """Extract points from the point cloud inside the given bounding box."""
        x_min, y_min, z_min = bbox[0]
        x_max, y_max, z_max = bbox[1]

        # Filter points inside the bounding box
        inside_bbox = (
            (pointcloud[:, 0] >= x_min) & (pointcloud[:, 0] <= x_max) &
            (pointcloud[:, 1] >= y_min) & (pointcloud[:, 1] <= y_max) &
            (pointcloud[:, 2] >= z_min) & (pointcloud[:, 2] <= z_max)
        )
        return pointcloud[inside_bbox]

    def _embed_pointcloud(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """Embeds the 3D point cloud."""
        pointcloud_embedding = self.pe_layer.forward_with_coords(pointcloud, self.input_pointcloud_size)
        return pointcloud_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds 3D mask inputs."""
        # print('masks:', masks.shape)
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(self) -> int:
        return self.batch_size

    def forward(self, bs: int, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds the 3D point cloud and bounding box proposals, returning both sparse and dense
        embeddings.

        Arguments:
          labels (torch.Tensor): Bounding boxes and point cloud data.
            - The first 3 columns represent the point cloud data (B, N, 3)
            - The fourth column contains the scalars (B, N, 1)

        Returns:
          torch.Tensor: sparse embeddings for the points, with shape (B, N, embed_dim)
          torch.Tensor: dense embeddings for the masks, with shape (B, embed_dim, H, W, D)
        """
        # print('labels:',labels.shape)
        # Extract pointcloud (first 3 columns) and scalars (fourth column)
        pointclouds = labels[:, :, :3]  # (B, N, 3)
        scalars = labels[:, :, 3]      # (B, N, 1)
        curvatures = labels[:, :, 4]
        surface_vars = labels[:, :, 5]

        # Initialize batch size and sparse_embeddings tensor
        self.batch_size = bs
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=pointclouds.device)

        # Voxelize pointclouds
        voxel_features = [self.voxelize(pc, scalar, curvature, surface_var) for pc, scalar, curvature, surface_var in zip(pointclouds, scalars, curvatures, surface_vars)]
        # print('voxel_features:',voxel_features[0].shape)

        # Embed the voxel_features into dense embeddings
        if voxel_features is not None:
            dense_embeddings = self._embed_masks(torch.stack(voxel_features, dim=0))
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
                pointclouds.shape[0], -1, self.input_pointcloud_size[0], self.input_pointcloud_size[1], self.input_pointcloud_size[2]
            )

        return sparse_embeddings, dense_embeddings

    def voxelize(
        self, 
        pointclouds: torch.Tensor, 
        scalars: torch.Tensor,
        curvature: torch.Tensor,
        surface_var: torch.Tensor,
        ) -> torch.Tensor:
        
        scalars = scalars.view(-1)  

        
        original_width, original_height = 1236, 1032  

        
        grid_x = torch.floor(scalars % original_width).long()  
        grid_y = torch.floor(scalars / original_width).long()  

        
        grid_x = torch.floor(grid_x * (self.input_pointcloud_size[0] - 1) / (original_width - 1)).long()
        grid_y = torch.floor(grid_y * (self.input_pointcloud_size[1] - 1) / (original_height - 1)).long()

        
        grid_z = torch.zeros_like(grid_x)  

        
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)

        
        voxel_features = torch.zeros(
            (self.input_pointcloud_size[0], self.input_pointcloud_size[1], 1), device=pointclouds.device
        )
        voxel_counts = torch.zeros_like(voxel_features)

        
        indices = grid[:, 0] * self.input_pointcloud_size[1] + grid[:, 1] 

        feature_all = curvature + surface_var

        
        voxel_features.view(-1)[indices] += feature_all
        voxel_counts.view(-1)[indices] += 1

        
        voxel_features = torch.where(
            voxel_counts.view(-1) > 0, voxel_features.view(-1) / voxel_counts.view(-1), torch.zeros_like(voxel_features.view(-1))
        )

        
        voxel_features = voxel_features.view(self.input_pointcloud_size[0], self.input_pointcloud_size[1], 1).unsqueeze(0).squeeze(-1)

        return voxel_features
