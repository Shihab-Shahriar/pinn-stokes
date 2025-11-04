"""Far-field Rotne-Prager-Yamakawa mobility tensor implemented in PyTorch."""

from __future__ import annotations

import importlib.util
import math
from contextlib import nullcontext
from pathlib import Path

from typing import Callable

import torch
import torch.nn.functional as F


def _trans_transpose(tensor: torch.Tensor) -> torch.Tensor:
	"""Return :math:`a_{jilk}` given :math:`a_{ijkl}`.

	Parameters
	----------
	tensor:
		Rank-4 tensor with indices ordered ``(i, j, k, l)``.
	"""

	return tensor.permute(1, 0, 3, 2)


def _epsilon_blocks(r_hat: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
	"""Build epsilon matrices for each bead pair scaled by ``scale``.

	Parameters
	----------
	r_hat:
		Tensor of shape ``(n, n, 3)`` containing unit displacement vectors.
	scale:
		Tensor of shape ``(n, n)`` with scalar prefactors.

	Returns
	-------
	torch.Tensor
		Tensor of shape ``(n, n, 3, 3)``.
	"""

	n = r_hat.shape[0]
	dtype = r_hat.dtype
	device = r_hat.device

	zeros = torch.zeros((n, n), dtype=dtype, device=device)
	x = r_hat[..., 0]
	y = r_hat[..., 1]
	z = r_hat[..., 2]

	row0 = torch.stack((zeros, scale * z, -scale * y), dim=-1)
	row1 = torch.stack((-scale * z, zeros, scale * x), dim=-1)
	row2 = torch.stack((scale * y, -scale * x, zeros), dim=-1)
	return torch.stack((row0, row1, row2), dim=-2)


def _epsilon_pair_blocks(r_hat: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
	"""Return epsilon matrices for a list of pairwise unit vectors.

	Parameters
	----------
	r_hat:
		Tensor of shape ``(m, 3)`` containing unit displacement vectors for each
		pair in the upper triangle.
	scale:
		Tensor of shape ``(m,)`` with scalar prefactors.

	Returns
	-------
	torch.Tensor
		Tensor of shape ``(m, 3, 3)``.
	"""

	zeros = torch.zeros_like(scale)
	x = r_hat[:, 0]
	y = r_hat[:, 1]
	z = r_hat[:, 2]

	row0 = torch.stack((zeros, scale * z, -scale * y), dim=-1)
	row1 = torch.stack((-scale * z, zeros, scale * x), dim=-1)
	row2 = torch.stack((scale * y, -scale * x, zeros), dim=-1)
	return torch.stack((row0, row1, row2), dim=-2)


def _reshape_blocks(block: torch.Tensor) -> torch.Tensor:
	"""Flatten an ``(n, n, 3, 3)`` tensor to ``(3n, 3n)`` preserving order."""

	n = block.shape[0]
	return block.permute(0, 2, 1, 3).reshape(n * 3, n * 3)


def _reshape_blocks_batch(block: torch.Tensor) -> torch.Tensor:
	"""Flatten a ``(batch, n, n, 3, 3)`` tensor to ``(batch, 3n, 3n)`` preserving order."""

	batch, n = block.shape[:2]
	return block.permute(0, 1, 3, 2, 4).reshape(batch, n * 3, n * 3)


def _flatten_blockmatrix(block: torch.Tensor) -> torch.Tensor:
	"""Flatten a ``(2, 2, n, n, 3, 3)`` block mobility tensor to ``(6n, 6n)``."""

	mu_tt = block[0, 0]
	mu_rt = block[0, 1]
	mu_tr = block[1, 0]
	mu_rr = block[1, 1]

	mu_tt_2d = _reshape_blocks(mu_tt)
	mu_rt_2d = _reshape_blocks(mu_rt)
	mu_tr_2d = _reshape_blocks(mu_tr)
	mu_rr_2d = _reshape_blocks(mu_rr)

	top = torch.cat((mu_tt_2d, mu_rt_2d), dim=1)
	bottom = torch.cat((mu_tr_2d, mu_rr_2d), dim=1)
	return torch.cat((top, bottom), dim=0)


def mu(
	centres: torch.Tensor,
	radii: torch.Tensor,
) -> torch.Tensor:
	"""Return the far-field grand mobility matrix flattened to ``(6n, 6n)``.

	The implementation assumes beads are well separated and omits
	self-interactions; diagonal blocks are set to zero.
	"""

	device = centres.device
	dtype = centres.dtype
	n = centres.shape[0]

	radii = radii.to(device=device, dtype=dtype)
	if n == 0:
		return torch.zeros((0, 0), dtype=dtype, device=device)
	if n == 1:
		return torch.zeros((6, 6), dtype=dtype, device=device)

	upper = torch.triu_indices(n, n, offset=1, device=device)
	i_idx = upper[0]
	j_idx = upper[1]
	n_pairs = i_idx.numel()

	diff = centres[i_idx] - centres[j_idx]
	distances = torch.linalg.norm(diff, dim=-1)
	r_hat = diff / distances.unsqueeze(-1)

	ai = radii[i_idx]
	aj = radii[j_idx]
	sum_a2 = ai ** 2 + aj ** 2

	pi = math.pi
	inv_r = 1.0 / distances
	inv_r2 = inv_r ** 2
	inv_r3 = inv_r ** 3

	aa = (1.0 / (8.0 * pi)) * inv_r
	bb = 1.0 + (sum_a2 * inv_r2) / 3.0
	TT_identity = aa * bb
	TT_rhat = aa * (1.0 - sum_a2 * inv_r2)

	I3 = torch.eye(3, dtype=dtype, device=device)
	r_hat_outer = r_hat[:, :, None] * r_hat[:, None, :]
	mu_tt_pairs = TT_identity[:, None, None] * I3 + TT_rhat[:, None, None] * r_hat_outer

	RR_identity = (-1.0 / (16.0 * pi)) * inv_r3
	RR_rhat = (3.0 / (16.0 * pi)) * inv_r3
	mu_rr_pairs = RR_identity[:, None, None] * I3 + RR_rhat[:, None, None] * r_hat_outer

	RT_scale = (1.0 / (8.0 * pi)) * inv_r2
	mu_rt_pairs = _epsilon_pair_blocks(r_hat, RT_scale)

	one_hot_i = F.one_hot(i_idx, num_classes=n).to(dtype=dtype)
	one_hot_j = F.one_hot(j_idx, num_classes=n).to(dtype=dtype)
	pair_mask = one_hot_i[:, :, None] * one_hot_j[:, None, :]
	pair_mask_T = pair_mask.transpose(-2, -1)

	upper_mask = pair_mask[..., None, None]
	lower_mask = pair_mask_T[..., None, None]

	mu_tt_upper = torch.sum(upper_mask * mu_tt_pairs[:, None, None, :, :], dim=0)
	mu_tt_lower = torch.sum(lower_mask * mu_tt_pairs[:, None, None, :, :].transpose(-1, -2), dim=0)
	mu_tt = mu_tt_upper + mu_tt_lower

	mu_rr_upper = torch.sum(upper_mask * mu_rr_pairs[:, None, None, :, :], dim=0)
	mu_rr_lower = torch.sum(lower_mask * mu_rr_pairs[:, None, None, :, :].transpose(-1, -2), dim=0)
	mu_rr = mu_rr_upper + mu_rr_lower

	mu_rt_upper = torch.sum(upper_mask * mu_rt_pairs[:, None, None, :, :], dim=0)
	mu_rt_lower = torch.sum(lower_mask * (-mu_rt_pairs)[:, None, None, :, :], dim=0)
	mu_rt = mu_rt_upper + mu_rt_lower

	mu_tr = _trans_transpose(mu_rt)

	mu_tt_2d = _reshape_blocks(mu_tt)
	mu_rt_2d = _reshape_blocks(mu_rt)
	mu_tr_2d = _reshape_blocks(mu_tr)
	mu_rr_2d = _reshape_blocks(mu_rr)

	top = torch.cat((mu_tt_2d, mu_rt_2d), dim=1)
	bottom = torch.cat((mu_tr_2d, mu_rr_2d), dim=1)
	return torch.cat((top, bottom), dim=0)


def _two_body_mu_batch(
	centres: torch.Tensor,
	radii: torch.Tensor,
) -> torch.Tensor:
	"""Compute batched two-body grand mobility matrices without vmap.

	Parameters
	----------
	centres:
		Tensor of shape ``(batch, 2, 3)`` containing bead centres.
	radii:
		Tensor of shape ``(batch, 2)`` with bead radii.

	Returns
	-------
	torch.Tensor
		Tensor of shape ``(batch, 12, 12)`` corresponding to flattened
		grand mobility matrices for each configuration.
	"""

	if centres.ndim != 3 or centres.shape[1] != 2 or centres.shape[2] != 3:
		raise ValueError("centres must have shape (batch, 2, 3)")
	if radii.ndim != 2 or radii.shape[1] != 2:
		raise ValueError("radii must have shape (batch, 2)")
	if centres.shape[0] != radii.shape[0]:
		raise ValueError("centres and radii batch dimensions must match")

	device = centres.device
	dtype = centres.dtype
	batch = centres.shape[0]

	if batch == 0:
		return torch.zeros((0, 12, 12), dtype=dtype, device=device)

	diff = centres[:, 0, :] - centres[:, 1, :]
	distances = torch.linalg.norm(diff, dim=-1)
	r_hat = diff / distances.unsqueeze(-1)

	a0 = radii[:, 0]
	a1 = radii[:, 1]
	sum_a2 = a0 ** 2 + a1 ** 2

	pi = math.pi
	inv_r = 1.0 / distances
	inv_r2 = inv_r ** 2
	inv_r3 = inv_r ** 3

	aa = (1.0 / (8.0 * pi)) * inv_r
	bb = 1.0 + (sum_a2 * inv_r2) / 3.0
	TT_identity = aa * bb
	TT_rhat = aa * (1.0 - sum_a2 * inv_r2)

	I3 = torch.eye(3, dtype=dtype, device=device)
	r_hat_outer = r_hat[:, :, None] * r_hat[:, None, :]
	mu_tt_pairs = TT_identity[:, None, None] * I3 + TT_rhat[:, None, None] * r_hat_outer

	RR_identity = (-1.0 / (16.0 * pi)) * inv_r3
	RR_rhat = (3.0 / (16.0 * pi)) * inv_r3
	mu_rr_pairs = RR_identity[:, None, None] * I3 + RR_rhat[:, None, None] * r_hat_outer

	RT_scale = (1.0 / (8.0 * pi)) * inv_r2
	mu_rt_pairs = _epsilon_pair_blocks(r_hat, RT_scale)

	mu_tt = torch.zeros((batch, 2, 2, 3, 3), dtype=dtype, device=device)
	mu_rr = torch.zeros_like(mu_tt)
	mu_rt = torch.zeros_like(mu_tt)

	mu_tt[:, 0, 1] = mu_tt_pairs
	mu_tt[:, 1, 0] = mu_tt_pairs.transpose(-1, -2)

	mu_rr[:, 0, 1] = mu_rr_pairs
	mu_rr[:, 1, 0] = mu_rr_pairs.transpose(-1, -2)

	mu_rt[:, 0, 1] = mu_rt_pairs
	mu_rt[:, 1, 0] = -mu_rt_pairs

	mu_tr = mu_rt.transpose(1, 2).transpose(-1, -2)

	mu_tt_2d = _reshape_blocks_batch(mu_tt)
	mu_rt_2d = _reshape_blocks_batch(mu_rt)
	mu_tr_2d = _reshape_blocks_batch(mu_tr)
	mu_rr_2d = _reshape_blocks_batch(mu_rr)

	top = torch.cat((mu_tt_2d, mu_rt_2d), dim=2)
	bottom = torch.cat((mu_tr_2d, mu_rr_2d), dim=2)
	return torch.cat((top, bottom), dim=1)


__all__ = ["mu", "accuracy_test", "measure_throughput", "bench"]


_REFERENCE_MU = None
_COMPILED_TWO_BODY: dict[torch.dtype, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}


def _get_reference_mu():
	global _REFERENCE_MU
	if _REFERENCE_MU is not None:
		return _REFERENCE_MU

	module_path = Path.home() / "hignn" / "grpy_tensors.py"
	if not module_path.exists():
		raise FileNotFoundError(f"Reference implementation not found at {module_path}")

	spec = importlib.util.spec_from_file_location("grpy_reference", module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Unable to load module specification from {module_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	_REFERENCE_MU = module.mu
	return _REFERENCE_MU


def _get_two_body_function(dtype: torch.dtype, use_compile: bool) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
	if not use_compile:
		return _two_body_mu_batch
	if not hasattr(torch, "compile"):
		return _two_body_mu_batch
	compiled = _COMPILED_TWO_BODY.get(dtype)
	if compiled is None:
		compiled = torch.compile(_two_body_mu_batch, dynamic=True)
		_COMPILED_TWO_BODY[dtype] = compiled
	return compiled


def accuracy_test() -> None:
	"""Compare torch implementation against reference NumPy version."""

	reference_mu = _get_reference_mu()
	device = torch.device("cpu")
	dtype = torch.float64

	test_cases = [
		(
			[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
			[0.3, 0.4],
		),
		(
			[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.5, 1.3, 0.0]],
			[0.2, 0.25, 0.3],
		),
		(
			[
				[0.0, 0.0, 0.0],
				[1.1, 0.4, 0.2],
				[0.3, 1.4, 0.6],
				[1.5, 1.3, 1.1],
			],
			[0.15, 0.18, 0.17, 0.16],
		),
	]

	for idx, (centres_list, radii_list) in enumerate(test_cases):
		centres = torch.tensor(centres_list, dtype=dtype, device=device)
		radii = torch.tensor(radii_list, dtype=dtype, device=device)

		displacements = centres[:, None, :] - centres[None, :, :]
		distances = torch.linalg.norm(displacements, dim=-1)
		ai = radii[:, None]
		aj = radii[None, :]
		mask = ~torch.eye(centres.shape[0], dtype=torch.bool, device=device)
		assert torch.all(distances[mask] > (ai + aj)[mask]), "Test case violates far-field assumption"

		ours = mu(centres, radii).cpu().to(dtype)
		reference_np = reference_mu(
			centres.cpu().numpy(),
			radii.cpu().numpy(),
			blockmatrix=True,
		)
		reference = torch.from_numpy(reference_np).to(dtype)
		reference = _flatten_blockmatrix(reference)

		n = centres.shape[0]
		ours = ours.clone()
		reference = reference.clone()
		for bead in range(n):
			start = bead * 6
			end = start + 6
			ours[start:end, start:end] = 0.0
			reference[start:end, start:end] = 0.0

		torch.testing.assert_close(
			ours,
			reference,
			rtol=1e-6,
			atol=1e-8,
		)


def _self_check() -> None:
	"""Run a lightweight correctness check on CPU or GPU."""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dtype = torch.float64

	centres = torch.tensor(
		[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.0, 0.0]],
		dtype=dtype,
		device=device,
	)
	radii = torch.tensor([1.0, 1.1, 0.9], dtype=dtype, device=device)

	displacements = centres[:, None, :] - centres[None, :, :]
	distances = torch.linalg.norm(displacements, dim=-1)
	ai = radii[:, None]
	aj = radii[None, :]
	mask = ~torch.eye(centres.shape[0], dtype=torch.bool, device=device)
	assert torch.all(distances[mask] > (ai + aj)[mask]), "Input violates far-field assumption"

	full = mu(centres, radii)

	torch.testing.assert_close(full, full.T, rtol=1e-6, atol=1e-8)
	diag = torch.diagonal(full)
	torch.testing.assert_close(diag, torch.zeros_like(diag))


def _generate_two_body_batch(
	batch_size: int,
	dtype: torch.dtype,
	device: torch.device,
	radius: float = 0.5,
	min_distance: float = 4.0,
	max_distance: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Sample separable two-body configurations respecting far-field limits."""

	if batch_size <= 0:
		raise ValueError("batch_size must be positive")
	if min_distance <= 2.0 * radius:
		raise ValueError("min_distance must exceed twice the bead radius")
	if max_distance <= min_distance:
		raise ValueError("max_distance must exceed min_distance")

	centres = torch.zeros((batch_size, 2, 3), dtype=dtype, device=device)
	radii = torch.full((batch_size, 2), radius, dtype=dtype, device=device)

	directions = torch.randn((batch_size, 3), dtype=dtype, device=device)
	norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
	directions = directions / norms.clamp_min(1e-12)
	distances = torch.empty((batch_size,), dtype=dtype, device=device)
	distances.uniform_(min_distance, max_distance)
	centres[:, 1, :] = directions * distances.unsqueeze(-1)
	return centres, radii


def measure_throughput(
	batch_size: int = 16384,
	n_warmup: int = 10,
	n_iter: int = 50,
	dtype: torch.dtype = torch.float32,
	use_amp: bool = False,
	amp_dtype: torch.dtype = torch.float16,
	use_compile: bool = False,
) -> float:
	"""Return two-body RPY evaluations per second on the active CUDA device.

	Parameters
	----------
	batch_size:
		Number of independent two-body problems evaluated per iteration.
	n_warmup:
		Number of warmup iterations before timing begins.
	n_iter:
		Number of timed iterations.
	dtype:
		Floating point precision used for the inputs and computation.
	use_amp:
		Enable ``torch.cuda.amp.autocast`` for the lookup when running on CUDA.
	amp_dtype:
		Target dtype for autocast (ignored when ``use_amp`` is ``False``).
	use_compile:
		Wrap ``mu`` with ``torch.compile(dynamic=True)`` for optimized execution.
	"""

	if not torch.cuda.is_available():
		raise RuntimeError("CUDA device is required to measure GPU throughput")
	if n_warmup < 0 or n_iter <= 0:
		raise ValueError("n_warmup must be >= 0 and n_iter must be > 0")
	if use_amp and not torch.cuda.is_available():
		raise RuntimeError("AMP requires a CUDA device")

	device = torch.device("cuda")
	centres, radii = _generate_two_body_batch(batch_size, dtype, device)
	mu_fn = _get_two_body_function(dtype, use_compile)

	if use_amp:
		def _amp_context():
			return torch.cuda.amp.autocast(dtype=amp_dtype)
	else:
		def _amp_context():
			return nullcontext()

	with torch.no_grad():
		for _ in range(n_warmup):
			with _amp_context():
				mu_fn(centres, radii)
	torch.cuda.synchronize()

	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)

	start.record()
	with torch.no_grad():
		for _ in range(n_iter):
			with _amp_context():
				outputs = mu_fn(centres, radii)
	end.record()
	torch.cuda.synchronize()

	duration_ms = start.elapsed_time(end)
	if duration_ms <= 0:
		return float("inf")
	elapsed_sec = duration_ms / 1000.0
	total_evaluations = batch_size * n_iter

	# Prevent elimination of the computation in extreme compiler settings.
	if outputs.numel() > 0:
		_ = float(outputs[0, 0, 0].item())

	return total_evaluations / elapsed_sec


def bench(
	batch_sizes: tuple[int, ...] | None = None,
	n_warmup: int = 10,
	n_iter: int = 50,
	dtype: torch.dtype = torch.float32,
	use_amp: bool = False,
	amp_dtype: torch.dtype = torch.float16,
	use_compile: bool = False,
) -> dict[int, float]:
	"""Run the throughput benchmark across batch sizes and print a summary."""

	if not torch.cuda.is_available():
		print("CUDA device not available; skipping throughput benchmark.")
		return {}
	if use_compile and not hasattr(torch, "compile"):
		print("torch.compile not available; falling back to eager execution.")
		use_compile = False

	if batch_sizes is None:
		batch_sizes = (512, 1024, 2048, 4096, 8192, 16384, 32768, 32768 * 2, 32768 * 4)

	print("\nRPY two-body throughput benchmark (GPU)")
	print(f"Timing {n_iter} iterations after {n_warmup} warmup runs.")
	precision_name = str(dtype).split(".")[-1]
	mode = "torch.compile" if use_compile else "eager"
	amp_note = f", amp={amp_dtype}" if use_amp else ""
	print(f"Precision: {precision_name}, mode: {mode}{amp_note}")

	results: dict[int, float] = {}
	for bs in batch_sizes:
		throughput = measure_throughput(
			bs,
			n_warmup=n_warmup,
			n_iter=n_iter,
			dtype=dtype,
			use_amp=use_amp,
			amp_dtype=amp_dtype,
			use_compile=use_compile,
		)
		results[bs] = throughput
		print(f"  batch={bs:>6} -> {throughput / 1e6:.2f} M eval/s")

	best_batch = max(results, key=results.get)
	best_throughput = results[best_batch]
	print(f"\nBest batch size: {best_batch} ({best_throughput / 1e6:.2f} M eval/s)")

	return results


if __name__ == "__main__":
	bench()

