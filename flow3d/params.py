import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.ops import knn_points
from loguru import logger as guru
from tqdm import tqdm
from cuml import KMeans

from flow3d.transforms import (
    cont_6d_to_rmat,
    rt_to_dq,
    dq_to_rt,
    rmat_to_cont_6d,
    normalize_dq,
    compute_relative_transform
)



class GaussianParams(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        scene_center: torch.Tensor | None = None,
        scene_scale: torch.Tensor | float = 1.0,
        visibilities: torch.Tensor | None = None,
    ):
        super().__init__()

        if not check_gaussian_sizes(means, quats, scales, colors, opacities):
            import ipdb

            ipdb.set_trace()

        params_dict = {
            "means": nn.Parameter(means),
            "quats": nn.Parameter(quats),
            "scales": nn.Parameter(scales),
            "colors": nn.Parameter(colors),
            "opacities": nn.Parameter(opacities),
        }
        self.params = nn.ParameterDict(params_dict)

        self.quat_activation = lambda x: F.normalize(x, dim=-1, p=2)
        self.color_activation = torch.sigmoid
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid

        if scene_center is None:
            scene_center = torch.zeros(3, device=means.device)
        self.register_buffer("scene_center", scene_center)
        self.register_buffer("scene_scale", torch.as_tensor(scene_scale))

        # Register visibilities as non-trainable buffer [G, T]
        # For foreground Gaussians: visibilities indicates if the Gaussian is visible at each frame
        # For background Gaussians: visibilities is None (not needed)
        if visibilities is not None:
            assert visibilities.shape[0] == means.shape[0], \
                f"visibilities shape mismatch: {visibilities.shape[0]} != {means.shape[0]}"
            self.register_buffer("visibilities", visibilities.float())
        else:
            self.register_buffer("visibilities", None)

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        req_keys = ["means", "quats", "scales", "colors", "opacities"]
        assert all(f"{prefix}{k}" in state_dict for k in req_keys)
        args = {
            "scene_center": torch.zeros(3),
            "scene_scale": torch.tensor(1.0),
            "visibilities": None,
        }
        for k in req_keys + list(args.keys()):
            if f"{prefix}{k}" in state_dict:
                args[k] = state_dict[f"{prefix}{k}"]

        return GaussianParams(**args)

    @property
    def num_gaussians(self) -> int:
        return self.params["means"].shape[0]

    def get_colors(self) -> torch.Tensor:
        return self.color_activation(self.params["colors"])

    def get_scales(self) -> torch.Tensor:
        return self.scale_activation(self.params["scales"])

    def get_opacities(self) -> torch.Tensor:
        return self.opacity_activation(self.params["opacities"])

    def get_quats(self) -> torch.Tensor:
        return self.quat_activation(self.params["quats"])

    def densify_params(self, should_split, should_dup):
        """
        densify gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_dup = x[should_dup]
            x_split = x[should_split].repeat([2] + [1] * (x.ndim - 1))

            if name == "scales":
                x_split -= math.log(1.6)

            x_new = nn.Parameter(torch.cat([x[~should_split], x_dup, x_split], dim=0))
            updated_params[name] = x_new
            self.params[name] = x_new

        # Propagate visibilities buffer with EXACT SAME INDEX ORDER
        if self.visibilities is not None:
            # 1. Keep non-split Gaussians
            vis_keep = self.visibilities[~should_split]
            # 2. Duplicate Gaussians
            vis_dup = self.visibilities[should_dup]
            # 3. Split Gaussians (each splits into 2, so repeat 2 times)
            vis_split = self.visibilities[should_split].repeat(2, 1)

            # Concatenate in the SAME ORDER: keep -> dup -> split
            self.visibilities = torch.cat([vis_keep, vis_dup, vis_split], dim=0)

            # Sanity check: ensure visibilities matches the new number of Gaussians
            assert self.visibilities.shape[0] == self.num_gaussians, \
                f"After densify: visibilities shape mismatch {self.visibilities.shape[0]} != {self.num_gaussians}"

        return updated_params

    def cull_params(self, should_cull):
        """
        cull gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_new = nn.Parameter(x[~should_cull])
            updated_params[name] = x_new
            self.params[name] = x_new

        # Cull visibilities with the SAME mask
        if self.visibilities is not None:
            self.visibilities = self.visibilities[~should_cull]

            # Sanity check: ensure visibilities matches the new number of Gaussians
            assert self.visibilities.shape[0] == self.num_gaussians, \
                f"After cull: visibilities shape mismatch {self.visibilities.shape[0]} != {self.num_gaussians}"

        return updated_params

    def reset_opacities(self, new_val):
        """
        reset all opacities to new_val
        """
        self.params["opacities"].data.fill_(new_val)
        updated_params = {"opacities": self.params["opacities"]}

        return updated_params


class MotionBasesPerLevel(nn.Module):
    def __init__(self, 
        rots: torch.Tensor, 
        transls: torch.Tensor,
    ):
        super().__init__()
        _, num_bases, num_frames, _ = rots.shape # (num_nodes_parent, K, T, 6)
        self.num_frames = num_frames
        self.num_bases = num_bases
        self.params = nn.ParameterDict(
            {
                "rots": nn.Parameter(rots), # (num_nodes_parent, K, T, 6)
                "transls": nn.Parameter(transls), # (num_nodes_parent, K, T, 3)
            }
        )


class MotionNodesPerLevel(nn.Module):
    def __init__(self, 
        positions: torch.Tensor,
        radius: torch.Tensor, 
        motion_coefs: torch.Tensor
    ):
        super().__init__()
        params_dict = {
            "positions": nn.Parameter(positions),  # position in cano_t (num_nodes, 3)
            "radius": nn.Parameter(radius),
            "motion_coefs": nn.Parameter(motion_coefs), # (num_nodes, K)
        }
        self.params = nn.ParameterDict(params_dict)
        self.scale_activation = torch.exp
        self.motion_coef_activation = lambda x: F.softmax(x, dim=-1)

    def get_positions(self) -> torch.Tensor:
        return self.params["positions"]

    def get_radius(self) -> torch.Tensor:
        return self.scale_activation(self.params["radius"])
    
    def get_coefs(self) -> torch.Tensor:
        return self.motion_coef_activation(self.params["motion_coefs"])
    
    @property
    def num_nodes(self) -> int:
        return self.get_positions().shape[0]
    
    def cull(self, should_cull):
        updated_params = {}
        for name, x in self.params.items():
            x_new = nn.Parameter(x[~should_cull])
            updated_params[name] = x_new
            self.params[name] = x_new

        return updated_params
    
    def add(self, param_dict):
        # param_dict: {"positions": xx, "radius": xx, "motion_coefs": xx}
        updated_params = {}
        for name, x in self.params.items():
            x_add = param_dict[name].to(x.device)
            x_new = nn.Parameter(torch.cat([x, x_add], dim=0))
            updated_params[name] = x_new
            self.params[name] = x_new

        return updated_params
    

class ParentIndicesPerLevel(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.set_indices(indices)

    def get_indices(self):
        return self.indices
    
    def set_indices(self, indices):
        self.register_buffer("indices", indices)

    def cull(self, cull_mask):
        # cull_mask: binary mask for culling
        indices = self.indices[~cull_mask]
        output, inverse_indices = torch.unique(indices, return_inverse=True)
        indices_range = torch.arange(len(output), device=output.device)
        compact_indices = indices_range[inverse_indices]
        self.set_indices(compact_indices)
    
    def add(self, indices):
        self.set_indices(torch.cat([self.get_indices(), indices]))
        

class MotionTree(nn.Module):
    def __init__(self, 
            child_nodes_per_level: list,
            motion_bases_per_level: list,
        ) -> None:
        super().__init__()

        self.device = torch.device("cuda:0")

        # get num of levels
        assert len(child_nodes_per_level) == len(motion_bases_per_level) # should have same num of levels
        num_levels = len(child_nodes_per_level)
        self.num_levels = num_levels

        self.register_buffer("child_nodes_per_level", torch.as_tensor(child_nodes_per_level))
        self.register_buffer("motion_bases_per_level", torch.as_tensor(motion_bases_per_level))

        # current leaf node level
        self.leaf_level = 0

    @property
    def num_nodes(self) -> int:
        counter = 0
        for nodes in self.motion_nodes:
            counter += nodes.num_nodes
        return counter

    def set_init_tree(self, motion_bases, motion_nodes, parent_indices):
        self.num_frames = motion_bases[0].num_frames

        self.motion_bases = nn.ParameterList(motion_bases)
        self.motion_nodes = nn.ParameterList(motion_nodes)
        self.parent_indices = nn.ModuleList(parent_indices)


    def init_from_state_dict(state_dict, prefix="motion_tree."):
        child_nodes_per_level = state_dict[f"{prefix}child_nodes_per_level"].tolist()
        motion_bases_per_level = state_dict[f"{prefix}motion_bases_per_level"].tolist()
        motion_tree = MotionTree(child_nodes_per_level, motion_bases_per_level)

        all_motion_bases = []
        all_motion_nodes = []
        all_parent_indices = []
        for level in range(motion_tree.num_levels):
            # motion bases
            prefix_motion_bases = f"{prefix}motion_bases.{level}.params."
            if f"{prefix_motion_bases}rots" not in state_dict.keys():
                continue
            elif level > 0:
                motion_tree.leaf_level += 1
            rots = state_dict[f"{prefix_motion_bases}rots"]
            transls = state_dict[f"{prefix_motion_bases}transls"]
            motion_bases = MotionBasesPerLevel(rots, transls)

            # motion nodes
            prefix_motion_nodes = f"{prefix}motion_nodes.{level}.params."
            node_positions = state_dict[f"{prefix_motion_nodes}positions"]
            node_radius = state_dict[f"{prefix_motion_nodes}radius"]
            node_motion_coefs = state_dict[f"{prefix_motion_nodes}motion_coefs"]
            motion_nodes = MotionNodesPerLevel(node_positions, node_radius, node_motion_coefs)

            # parent indices
            indices = state_dict[f"{prefix}parent_indices.{level}.indices"]
            parent_indices = ParentIndicesPerLevel(indices)

            all_motion_bases.append(motion_bases)
            all_motion_nodes.append(motion_nodes)
            all_parent_indices.append(parent_indices)
        

        motion_tree.set_init_tree(all_motion_bases, all_motion_nodes, all_parent_indices)

        return motion_tree


    def compute_node_world_transforms(self, ts, level=None):
        if level is None:
            level = self.leaf_level

        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        
        transforms_parent = torch.eye(4)[None, None, ...].to(self.device) # (1, 1, 4, 4)
        for l in range(level+1):

            transforms = self.compute_per_level_node_local_transforms(ts, l)
            transforms = torch.cat((transforms, bottom.view(1, 1, 1, 4).expand_as(transforms[:, :, :1, :])), dim=-2) # (..., 4, 4)
            
            parent_indices = self.parent_indices[l].get_indices()

            # transforms in the world coodinate system
            transforms = transforms @ transforms_parent[parent_indices]
            transforms_parent = transforms

        return transforms[..., :3, :] # (num_nodes, t, 3, 4)


    def compute_per_level_node_local_transforms(self, ts, level=0):
        # motion bases
        transls = self.motion_bases[level].params["transls"][:, :, ts] # (num_nodes_parent, K, t, 3)
        rots = self.motion_bases[level].params["rots"][:, :, ts] # (num_nodes_parent, K, t, 6)

        # motion coefs
        coefs = self.motion_nodes[level].get_coefs() # (num_nodes, K), where K is the number of motion bases at level L

        # motion blending
        indices = self.parent_indices[level].get_indices() # (num_nodes, ), s.t. max(num_nodes) = num_nodes_parent - 1
        transls_scattered = transls[indices] # (num_nodes, K, t, 3)
        rots_scattered = rots[indices] # (num_nodes, K, t, 6)
        transls_nodes = (coefs[..., None, None] * transls_scattered).sum(1) # (num_nodes, t, 3)
        rots_nodes = (coefs[..., None, None] * rots_scattered).sum(1) # (num_nodes, t, 6)

        rotmats_nodes = cont_6d_to_rmat(rots_nodes)  # (num_nodes, t, 3, 3)

        transforms_nodes = torch.cat([rotmats_nodes, transls_nodes[..., None]], dim=-1) # (num_nodes, t, 3, 4)

        return transforms_nodes


    def compute_knn_nodes(self, means, k, level=None, no_softmax=False):
        if level is None:
            level = self.leaf_level

        # get positions and radius for the level
        positions = self.motion_nodes[level].get_positions()
        radius = self.motion_nodes[level].get_radius()

        inds, weights = self.compute_knn_inds_weights(means, k, positions, radius, no_softmax=no_softmax)

        return inds, weights
    
    @staticmethod
    def compute_knn_inds_weights(x, k, node_pos, node_radius, no_softmax=False):
        """ core function for knn node query and RBF weight computation

        Args:
            x (torch.Tensor): (N, 3)
            k (int): knn
            node_pos (torch.Tensor): (M, 3)
            node_radius (torch.Tensor): (M,)
        """
        nn_dist, nn_idxs, _ = knn_points(x[None], node_pos[None], None, None, K=k)
        dist, idxs = nn_dist[0], nn_idxs[0]
        weights = torch.exp(-(dist ** 2) / (2 * node_radius[idxs]**2 + 1e-8))  # #(G, k)

        if no_softmax:
            return idxs, weights

        return idxs, F.softmax(weights, dim=-1)


    def compute_transforms_from_nodes(self, ts, means, level=None):
        """ Compute Gaussians' transforms
        """
        if level is None:
            level = self.leaf_level
        
        knn_ids, knn_weights = self.compute_knn_nodes(means.detach(), k=3, level=level)  # [num_nodes, k]
        node_transforms = self.compute_node_world_transforms(ts, level=level) # (num_nodes, t, 3, 4)

        # dual quaternion blending (DQB)
        node_transforms_dq = rt_to_dq(node_transforms[..., :3, :3], node_transforms[..., :3, 3])
        gaussian_transforms_dq = (node_transforms_dq[knn_ids] * knn_weights[..., None, None]).sum(1) + 1e-8
        gaussian_transforms_dq = normalize_dq(gaussian_transforms_dq)
        rot, trans = dq_to_rt(gaussian_transforms_dq)
        transforms = torch.cat([rot, trans.unsqueeze(-1)], dim=-1)

        return transforms
    

    def _find_neighbor_gaussians(self, parent_node_positions, gaussian_means, level, num_max_gaussians=1000):
        """Find neighbor Gaussians for each parent node."""
        distance_node_to_gaussians = torch.norm(
            parent_node_positions[:, None, :] - gaussian_means[None, :, :], dim=-1
        )
        node_radius = self.motion_nodes[level].get_radius()
        masks = distance_node_to_gaussians < node_radius[:, None]
        sorted_masks, indices = (masks * torch.rand_like(masks, dtype=torch.float)).sort(dim=-1, descending=True)
        sorted_masks[:, num_max_gaussians:] = 0
        return (sorted_masks > 0).gather(-1, indices.argsort(-1))

    
    def _perform_kmeans_clustering(self, features, num_clusters):
        """ Perform K-means clustering on the features."""
        kmeans_model = KMeans(n_clusters=num_clusters)
        kmeans_model.fit(features)
        cluster_ids = torch.as_tensor(kmeans_model.labels_)
        if len(cluster_ids.unique()) != num_clusters:
            return None
        return cluster_ids
    
    def _compute_motion_bases(self, gaussian_means, gaussian_transforms, cluster_ids, num_bases):
        bases_rots, bases_transls = [], []
        cluster_centers = torch.stack([
            gaussian_means[(cluster_ids == j).clone()].median(dim=0).values for j in range(num_bases)
        ])
        for j in range(num_bases):
            mask = (cluster_ids == j)
            cluster_gaussian_positions = gaussian_means[mask]
            cluster_gaussian_transform = gaussian_transforms[mask]  
            cluster_gaussian_radius = torch.ones_like(cluster_gaussian_positions[..., 0])
            cluster_gaussian_radius = 0.05 * cluster_gaussian_radius
            knn_ids, knn_weights = self.compute_knn_inds_weights(cluster_centers[j][None], 10, cluster_gaussian_positions, cluster_gaussian_radius)
            cluster_gaussian_transforms_dq = rt_to_dq(cluster_gaussian_transform[..., :3, :3], cluster_gaussian_transform[..., :3, 3])
            center_transforms_dq = (cluster_gaussian_transforms_dq[knn_ids] * knn_weights[..., None, None]).sum(1) + 1e-8
            center_transforms_dq = normalize_dq(center_transforms_dq)
            rot, trans = dq_to_rt(center_transforms_dq)
            bases_rots.append(rmat_to_cont_6d(rot))
            bases_transls.append(trans)
        return torch.cat(bases_rots), torch.cat(bases_transls), cluster_centers
    
    def _compute_child_nodes(self, gaussian_means, cluster_centers, num_nodes):
        num_gaussians = len(gaussian_means)
        if num_gaussians < num_nodes:
            return None
        assert num_gaussians >= num_nodes
        sample_indices = torch.randperm(num_gaussians)[:num_nodes]
        node_positions = gaussian_means[sample_indices]
        nodes2centers = torch.norm(node_positions[:, None] - cluster_centers[None, :], dim=-1)
        node_motion_coefs = 10 * torch.exp(-nodes2centers)
        k = 3
        nn_dist, _, _ = knn_points(node_positions[None], node_positions[None], K=k+1, return_sorted=False)
        nn_dist = nn_dist[0]
        radius = nn_dist[..., 1:].mean(dim=-1, keepdim=False)
        radius = radius.clamp(
            torch.quantile(radius, 0.05), torch.quantile(radius, 0.95)
        )
        radius = torch.log(radius)
        node_radius = radius
        return node_positions, node_motion_coefs, node_radius
            
    
    def _process_nodes(self, node_transforms, gaussian_means, masks, num_bases, num_nodes, ts, level):
        """Process each node to compute motion bases, nodes, and parent indices"""
        bases_rots_all, bases_transls_all = [], []
        node_positions_all, node_motion_coefs_all, node_radius_all = [], [], []
        parent_indices_all = []
        parent_indices_to_remove = []
        parent_index = 0

        for i in tqdm(range(len(node_transforms)), desc="Adding next level nodes/bases"):
            if masks[i].sum() == 0:
                guru.warning(f"No valid Gaussians found for node {i}.")
                parent_indices_to_remove.append(i)
                continue

            # Process valid Gaussians for the current node
            result = self._process_single_node(
                i, node_transforms[i], gaussian_means, masks[i], num_bases, num_nodes, ts, level, parent_index
            )

            if result is None:
                parent_indices_to_remove.append(i)
                continue

            # Unpack results
            (
                bases_rots, bases_transls, node_positions, node_motion_coefs, node_radius, parent_indices
            ) = result

            bases_rots_all.append(bases_rots)
            bases_transls_all.append(bases_transls)
            node_positions_all.append(node_positions)
            node_motion_coefs_all.append(node_motion_coefs)
            node_radius_all.append(node_radius)
            parent_indices_all.append(parent_indices)
            parent_index += 1
        
        return {
            "bases_rots_all": bases_rots_all,
            "bases_transls_all": bases_transls_all,
            "node_positions_all": node_positions_all,
            "node_motion_coefs_all": node_motion_coefs_all,
            "node_radius_all": node_radius_all,
            "parent_indices_all": parent_indices_all,
            "parent_indices_to_remove": parent_indices_to_remove,
        }
            

    def _process_single_node(self, node_idx, node_transform, gaussian_means, mask, num_bases, num_nodes, ts, level, parent_index):
        """Process a single node to compute motion baes and child nodes."""
        valid_gaussian_means = gaussian_means[mask]
        valid_gaussian_transforms = self.compute_transforms_from_nodes(ts, valid_gaussian_means, level=level)

        # Compute relative transforms
        valid_gaussian_relative_transform = compute_relative_transform(
            node_transform[None], valid_gaussian_transforms
        )
        relative_positions = valid_gaussian_relative_transform[..., :3, 3]
        velocities = relative_positions[:, 1:] - relative_positions[:, :-1]
        velocity_dirs = F.normalize(velocities, p=2, dim=-1)
        valid_gaussian_features = velocity_dirs.view(len(velocity_dirs), -1)

        # Perform K-means clustering
        cluster_ids = self._perform_kmeans_clustering(valid_gaussian_features, num_bases)
        if cluster_ids is None:
            guru.warning(f"K-means clustering failed for node {node_idx}. Delete this node")
            return None
        
        # Comput motion bases
        bases_rots, bases_transls, cluster_centers = self._compute_motion_bases(
            valid_gaussian_means, valid_gaussian_relative_transform, cluster_ids, num_bases
        )
        nodes_results = self._compute_child_nodes(
            valid_gaussian_means, cluster_centers, num_nodes
        )
        if nodes_results is None:
            guru.warning(f">>>>>>> num gaussians error!!!!")
            return None
        node_positions, node_motion_coefs, node_radius = nodes_results
        parent_indices = torch.ones(num_nodes, dtype=torch.long, device=self.device) * parent_index


        return bases_rots, bases_transls, node_positions, node_motion_coefs, node_radius, parent_indices       


    def _update_tree_with_new_level(self, results):
        motion_nodes_deeper = MotionNodesPerLevel(
            torch.cat(results["node_positions_all"]),
            torch.cat(results["node_radius_all"]),
            torch.cat(results["node_motion_coefs_all"]),
        )
        motion_bases_deeper = MotionBasesPerLevel(
            torch.stack(results["bases_rots_all"]),
            torch.stack(results["bases_transls_all"]),
        )
        parent_indices_deeper = ParentIndicesPerLevel(torch.cat(results["parent_indices_all"]))

        self.motion_nodes.append(motion_nodes_deeper)
        self.motion_bases.append(motion_bases_deeper)
        self.parent_indices.append(parent_indices_deeper)    


    @torch.no_grad()
    def deepen(self, gaussian_means, level=None):
        """Core function to add one more level to the motion tree.

        Args:
            gaussian_meand (torch.Tensor): Gaussian menas (num_gaussians, 3).
            level (int, optional): Current level of the tree. Defaults to None.
        
        Returns:
            list: Indices of parent nodes to remove.
        """
        if level is None:
            level = self.leaf_level
        
        if level == self.num_levels - 1:
            guru.info("Already reached the leaf node!")
            return 
        
        parent_node_positions = self.motion_nodes[level].get_positions()
        num_bases = self.motion_bases_per_level[level + 1]
        num_nodes = self.child_nodes_per_level[level + 1]

        # Step 1: Find neighbor Gaussians
        masks = self._find_neighbor_gaussians(parent_node_positions, gaussian_means, level)

        # Step 2: Compute all node transforms
        ts = torch.arange(self.num_frames, device=self.device)
        node_transforms = self.compute_node_world_transforms(ts, level=level)

        # Step 3: Process each node
        results = self._process_nodes(
            node_transforms, gaussian_means, masks, num_bases, num_nodes, ts, level
        )
        
        # Step 4: Stack results and update tree
        self._update_tree_with_new_level(results)

        guru.info(f"Deepen motion tree: level {self.leaf_level} => {self.leaf_level + 1}")
        self.leaf_level += 1

        return results["parent_indices_to_remove"]


def check_gaussian_sizes(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    motion_coefs: torch.Tensor | None = None,
) -> bool:
    dims = means.shape[:-1]
    leading_dims_match = (
        quats.shape[:-1] == dims
        and scales.shape[:-1] == dims
        and colors.shape[:-1] == dims
        and opacities.shape == dims
    )
    if motion_coefs is not None and motion_coefs.numel() > 0:
        leading_dims_match &= motion_coefs.shape[:-1] == dims
    dims_correct = (
        means.shape[-1] == 3
        and (quats.shape[-1] == 4)
        and (scales.shape[-1] == 3)
        and (colors.shape[-1] == 3)
    )
    return leading_dims_match and dims_correct