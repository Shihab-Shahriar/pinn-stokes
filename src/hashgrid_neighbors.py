from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import warp as wp

wp.init()


@wp.kernel
def count_neighbors_kernel(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    no_of_nn: wp.array(dtype=wp.uint8),
    radius: float,
    radius_sq: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    x = positions[i]
    count = wp.uint8(0)

    neighbors = wp.hash_grid_query(grid, x, radius)

    for index in neighbors:
        if index != i:
            n = x - positions[index]
            d = wp.length_sq(n)
            if d < radius_sq:
                count += wp.uint8(1)
    no_of_nn[i] = count


@wp.kernel
def fill_edge_indexes_kernel(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    edge_t: wp.array(dtype=wp.int32),
    edge_s: wp.array(dtype=wp.int32),
    idx_start: wp.array(dtype=wp.int32),
    radius: float,
    radius_sq: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    x = positions[i]
    idx = idx_start[i]

    neighbors = wp.hash_grid_query(grid, x, radius)

    for index in neighbors:
        if index != i:
            d = wp.length_sq(x - positions[index])
            if d < radius_sq:
                edge_t[idx] = i
                edge_s[idx] = index
                idx += 1


class HashGridNeighborSearch:
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        grid_dim: Tuple[int, int, int] = (128, 128, 128),
        max_pairs: Optional[int] = None,
    ) -> None:
        self.device = torch.device(device)
        self.device_str = str(self.device)
        self.grid = wp.HashGrid(*grid_dim, device=self.device_str)
        self.no_of_nn = None   # WARNING: wp.uint8 array. max 255.

        self.edge_indexes_s = None
        self.edge_indexes_t = None
        self._capacity = 0
        self._max_pairs = int(max_pairs) if max_pairs is not None else None
        if self._max_pairs is not None:
            self._allocate_edge_indexes(self._max_pairs, self.device)

    def _allocate_edge_indexes(self, size: int, device: torch.device) -> None:
        self.edge_indexes_s = torch.empty((size,), dtype=torch.int32, device=device)
        self.edge_indexes_t = torch.empty((size,), dtype=torch.int32, device=device)
        self._capacity = size

    def _ensure_edge_capacity(self, total_pairs: int, device: torch.device) -> None:
        if total_pairs <= 0:
            return
        if (
            self.edge_indexes_t is not None
            and self.edge_indexes_t.device == device
            and self.edge_indexes_t.numel() >= total_pairs
        ):
            return

        if self._max_pairs is not None and total_pairs > self._max_pairs:
            raise AssertionError(
                f"Too many near-field pairs ({total_pairs}); increase max_pairs in HashGridNeighborSearch"
            )

        if self._max_pairs is not None:
            new_size = self._max_pairs
        else:
            new_size = max(total_pairs, self._capacity * 2 if self._capacity else total_pairs)
        self._allocate_edge_indexes(new_size, device)

    def get_edge_indexes(
        self,
        positions_t: torch.Tensor,
        radius: float,
        wp_stream: Optional[wp.Stream] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not positions_t.is_cuda:
            raise AssertionError("positions_t must be a CUDA tensor for HashGrid neighbor search")

        torch.cuda.synchronize()
        positions_t = positions_t.contiguous()
        radius = float(radius)
        radius_sq = radius * radius

        N = positions_t.size(0)
        if N == 0:
            empty = torch.empty((0,), dtype=torch.int32, device=positions_t.device)
            return empty, empty

        if wp_stream is None:
            torch_stream = torch.cuda.current_stream(device=positions_t.device)
            wp_stream = wp.stream_from_torch(torch_stream)

        with wp.ScopedStream(wp_stream):
            if self.no_of_nn is None or self.no_of_nn.shape[0] != N:
                self.no_of_nn = wp.zeros(
                    shape=[N],
                    dtype=wp.uint8,
                    device=str(positions_t.device),
                    requires_grad=False,
                )

            p = wp.from_torch(positions_t, dtype=wp.vec3)
            if verbose:
                print("Building spatial grid for neighbor search...")
                print("Positions shape:", p.shape)
            self.grid.build(points=p, radius=radius)

            if verbose:
                print("Counting neighbors per particle...")
            wp.launch(
                kernel=count_neighbors_kernel,
                dim=N,
                inputs=(self.grid.id, p, self.no_of_nn, radius, radius_sq),
                stream=wp_stream,
            )
            wp.synchronize()

            nn_count_torch_uint8 = wp.to_torch(self.no_of_nn)
            nn_count_torch = torch.cumsum(nn_count_torch_uint8, dim=0, dtype=torch.int32)
            total_pairs = int(nn_count_torch[-1].item())
            if verbose:
                print(f"Total near-field pairs found: {total_pairs}")

            if total_pairs == 0:
                if self.edge_indexes_t is not None and self.edge_indexes_t.device == positions_t.device:
                    return self.edge_indexes_t[:0], self.edge_indexes_s[:0]
                empty = torch.empty((0,), dtype=torch.int32, device=positions_t.device)
                return empty, empty

            self._ensure_edge_capacity(total_pairs, positions_t.device)

            nn_count_torch = nn_count_torch - nn_count_torch_uint8
            if verbose:
                print("Collecting neighbor edge indexes...")
            wp_edges_t = wp.from_torch(self.edge_indexes_t, dtype=wp.int32)
            wp_edges_s = wp.from_torch(self.edge_indexes_s, dtype=wp.int32)
            wp_idx_start = wp.from_torch(nn_count_torch, dtype=wp.int32)
            wp.launch(
                kernel=fill_edge_indexes_kernel,
                dim=N,
                inputs=(self.grid.id, p, wp_edges_t, wp_edges_s, wp_idx_start, radius, radius_sq),
                stream=wp_stream,
            )
            wp.synchronize()

        return self.edge_indexes_t[:total_pairs], self.edge_indexes_s[:total_pairs]
