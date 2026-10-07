# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Shared data structures, tiling specs, and cost models for mHC-lite Pallas kernels."""

import contextlib
import dataclasses
import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


@contextlib.contextmanager
def tpu_mesh_context():
  """Ensures a valid TPU AbstractMesh is available when running on CPU / interpret mode."""
  needs_abstract_mesh = False
  try:
    if not pltpu.is_tpu_device():
      needs_abstract_mesh = True
  except (RuntimeError, ValueError, AttributeError, TypeError):
    needs_abstract_mesh = True

  if needs_abstract_mesh:
    abstract_mesh = jax.sharding.AbstractMesh(
        (),
        (),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      yield
  else:
    yield


HBM_SPEC = pl.BlockSpec(memory_space=pltpu.HBM)
# Kernel-level context tuple: `(x, h_post, residual)`.
type KernelContext = tuple[jax.Array, jax.Array, jax.Array]


class UnsupportedInputError(ValueError):
  """Known Mosaic shape, dtype, or tiling restriction."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MHCContext:
  """Opaque token-local context passed from `pre` to `post`."""

  x: jax.Array
  h_post: jax.Array
  residual: jax.Array
  implementation: str = dataclasses.field(metadata={"static": True})


@dataclasses.dataclass(frozen=True)
class MhcKernelConfig:
  """Compiler and tiling configuration for mHC Pallas kernels."""

  block_size: int = 128
  fwd_pipeline_blocks: int = 2
  bwd_block_size: int = 256
  bwd_feature_block_size: int = 1024
  vmem_limit_bytes: int = 128 * 1024 * 1024
  rms_epsilon: float = 1e-5
  pre_mapping_epsilon: float = 1e-6
  interpret: bool = False


@dataclasses.dataclass(frozen=True)
class MhcDims:
  """Static dimension and cost descriptor for mHC-lite."""

  tokens: int
  streams: int
  embedding: int
  num_permutations: int = 24

  @property
  def flattened_size(self) -> int:
    return self.streams * self.embedding

  @property
  def pre_slice(self) -> slice:
    return slice(0, self.streams)

  @property
  def post_slice(self) -> slice:
    return slice(self.streams, 2 * self.streams)

  @property
  def res_slice(self) -> slice:
    return slice(2 * self.streams, 2 * self.streams + self.num_permutations)

  @property
  def phi_cols(self) -> int:
    return 2 * self.streams + self.num_permutations

  def coeff_fwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the forward coefficient kernel.

    Mathematical Derivations:
      - FLOPs:
        1. Fused Projection GEMM: `flattened_x (T, k*d) @ phi (k*d, 2*k+P)`
           [einsum: `tf,fc->tc` where `f = k*d`, `c = 2*k + P`]
           = `2 * T * (k*d) * (2*k + P)` FLOPs (2 FLOPs per
           multiply-accumulate).
        2. Permutation GEMM: `softmax_weights (T, P) @ permutations (P, k*k)`
           [einsum: `tp,pij->tij` where `i, j` are streams `k, k`]
           = `2 * T * P * k*k` FLOPs.
        Total FLOPs = `2 * tokens * flattened_size * phi_cols + 2 * tokens *
        num_permutations * streams^2`.
      - Transcendentals:
        Sigmoid gating for pre/post gates and softmax over permutation logits
        = `T * (k + k + P) = tokens * (2 * streams + num_permutations) = tokens
        * phi_cols`.
      - Bytes Accessed:
        Read input `x` (bfloat16: `T * k * d * 2` bytes) + read `phi` (float32:
        `(k*d) * phi_cols * 4` bytes)
        = `tokens * streams * embedding * 2 + flattened_size * phi_cols * 4`
        bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(
        2 * self.tokens * self.flattened_size * self.phi_cols
        + 2 * self.tokens * self.num_permutations * self.streams * self.streams
    )
    transcendentals = int(
        self.tokens * (2 * self.streams + self.num_permutations)
    )
    bytes_accessed = int(
        self.tokens * self.streams * self.embedding * 2
        + self.flattened_size * self.phi_cols * 4
    )
    return pl.CostEstimate(
        flops=flops,
        transcendentals=transcendentals,
        bytes_accessed=bytes_accessed,
    )

  def pre_apply_fwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the forward pre-apply gating kernel.

    Mathematical Derivations:
      - FLOPs:
        Stream reduction `sum_s (h_pre[t, s] * x[t, s, d])` [einsum:
        `ts,tsd->td`] across `k` streams
        for each of the `T * d` output elements (vector-matrix product `(1, k) @
        (k, d) -> (1, d)` per token).
        = `2 * T * k * d = 2 * tokens * streams * embedding` FLOPs.
      - Transcendentals:
        0 (linear scaling and summation).
      - Bytes Accessed:
        Read input `x` (bfloat16: `T * k * d * 2` bytes) + write `layer_input`
        (bfloat16: `T * d * 2` bytes)
        = `tokens * streams * embedding * 2 + tokens * embedding * 2` bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(2 * self.tokens * self.streams * self.embedding)
    bytes_accessed = int(
        self.tokens * self.streams * self.embedding * 2
        + self.tokens * self.embedding * 2
    )
    return pl.CostEstimate(
        flops=flops, transcendentals=0, bytes_accessed=bytes_accessed
    )

  def post_apply_fwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the forward post-apply mixing kernel.

    Mathematical Derivations:
      - FLOPs:
        1. Residual mixing contraction `sum_s_in (residual[t, s_in, s_out] *
        x[t, s_in, d])`
           [einsum: `tkj,tkd->tjd`] (matrix product `(k, k) @ (k, d) -> (k, d)`
           per token):
           `2 * k` FLOPs per output element over `T * k * d` elements = `2 * T *
           k^2 * d` FLOPs.
        2. Post-gating and accumulation `h_post[t, s] * layer_output[t, d] +
        residual_mix[t, s, d]`
           [einsum: `ts,td->tsd`]:
           `2 * T * k * d` FLOPs (1 multiply + 1 add per element).
        Total FLOPs = `2 * tokens * streams^2 * embedding + 2 * tokens * streams
        * embedding`.
      - Transcendentals:
        0.
      - Bytes Accessed:
        Read `x` (bfloat16: `T * k * d * 2` bytes) + write output (bfloat16: `T
        * k * d * 2` bytes)
        + read `layer_output` and gating/residual context (`T * d * 4` bytes)
        = `2 * tokens * streams * embedding * 2 + tokens * embedding * 4` bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(
        2 * self.tokens * self.streams * self.streams * self.embedding
        + 2 * self.tokens * self.streams * self.embedding
    )
    bytes_accessed = int(
        2 * self.tokens * self.streams * self.embedding * 2
        + self.tokens * self.embedding * 4
    )
    return pl.CostEstimate(
        flops=flops, transcendentals=0, bytes_accessed=bytes_accessed
    )

  def pre_apply_bwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the backward pre-apply kernel.

    Mathematical Derivations:
      - FLOPs:
        1. Activation cotangent `d_x = h_pre * d_layer_input` [einsum:
        `ts,td->tsd`] (multiply)
           + accumulation with `d_x_acc` (add) = `2 * T * k * d` FLOPs.
        2. Gate cotangent `d_h_pre = sum_d (x * d_layer_input)` [einsum:
        `tsd,td->ts`] (multiply + reduce-add)
           = `2 * T * k * d` FLOPs.
        Total FLOPs = `4 * tokens * streams * embedding`.
      - Transcendentals:
        0.
      - Bytes Accessed:
        Read `x` (`T * k * d * 2`) + read `d_x_acc` (`T * k * d * 2`) + write
        `d_x_acc_out` (`T * k * d * 2`)
        + read `d_layer_input` (`T * d * 2`)
        = `3 * tokens * streams * embedding * 2 + tokens * embedding * 2` bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(4 * self.tokens * self.streams * self.embedding)
    bytes_accessed = int(
        3 * self.tokens * self.streams * self.embedding * 2
        + self.tokens * self.embedding * 2
    )
    return pl.CostEstimate(
        flops=flops, transcendentals=0, bytes_accessed=bytes_accessed
    )

  def coeff_bwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the backward coefficient kernel.

    Mathematical Derivations:
      - FLOPs:
        In-kernel forward recomputation plus VJP of `mhc_coeffs` (evaluating
        matrix multiplications
        for activation cotangents `d_x` [einsum: `tc,fc->tf`], parameter
        gradients `d_phi` [einsum: `tf,tc->fc`],
        and permutation gradients [einsum: `tij,pij->tp`]), scaling forward
        GEMMs by 2x:
        Total FLOPs = `2 * (2 * tokens * flattened_size * phi_cols + 2 * tokens
        * num_permutations * streams^2)`.
      - Transcendentals:
        Gating and softmax evaluations across tokens
        = `T * (k + k + P) = tokens * (2 * streams + num_permutations)`.
      - Bytes Accessed:
        Read `x` (`T * k * d * 2`) + read `d_x_acc` (`T * k * d * 2`) + write
        `d_x` (`T * k * d * 2`)
        + read `phi` and write `d_phi` (`2 * (k*d) * phi_cols * 4`)
        = `3 * tokens * streams * embedding * 2 + 2 * flattened_size * phi_cols
        * 4` bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(
        2
        * (
            2 * self.tokens * self.flattened_size * self.phi_cols
            + 2
            * self.tokens
            * self.num_permutations
            * self.streams
            * self.streams
        )
    )
    transcendentals = int(
        self.tokens * (2 * self.streams + self.num_permutations)
    )
    bytes_accessed = int(
        3 * self.tokens * self.streams * self.embedding * 2
        + 2 * self.flattened_size * self.phi_cols * 4
    )
    return pl.CostEstimate(
        flops=flops,
        transcendentals=transcendentals,
        bytes_accessed=bytes_accessed,
    )

  def post_apply_bwd_cost(self) -> pl.CostEstimate:
    """Estimates compute and memory cost for the backward post-apply kernel.

    Mathematical Derivations:
      - FLOPs:
        Backward VJP of post-apply evaluates transposed residual contraction
        `d_x = residual^T @ d_output`
        [einsum: `tkj,tjd->tkd`] (`2 * T * k^2 * d`), gate reduction
        `d_layer_output = sum_s (h_post * d_output)`
        [einsum: `ts,tsd->td`] (`2 * T * k * d`), and feature reductions
        `d_h_post` [einsum: `td,tsd->ts`]
        (`2 * T * k * d`) and `d_residual` [einsum: `tjd,tkd->tjk`] (`2 * T *
        k^2 * d`):
        Total FLOPs = `4 * tokens * streams^2 * embedding + 4 * tokens * streams
        * embedding`
        = `2 * (2 * tokens * streams^2 * embedding + 2 * tokens * streams *
        embedding)`.
      - Transcendentals:
        0.
      - Bytes Accessed:
        Read `x` (`T * k * d * 2`) + read `d_output` (`T * k * d * 2`) + write
        `d_x` (`T * k * d * 2`)
        + read/write `layer_output` and `d_layer_output` (`2 * tokens *
        embedding * 4`)
        = `3 * tokens * streams * embedding * 2 + 2 * tokens * embedding * 4`
        bytes.

    Returns:
      pl.CostEstimate with estimated FLOPs, transcendentals, and bytes accessed.
    """
    flops = int(
        2
        * (
            2 * self.tokens * self.streams * self.streams * self.embedding
            + 2 * self.tokens * self.streams * self.embedding
        )
    )
    bytes_accessed = int(
        3 * self.tokens * self.streams * self.embedding * 2
        + 2 * self.tokens * self.embedding * 4
    )
    return pl.CostEstimate(
        flops=flops, transcendentals=0, bytes_accessed=bytes_accessed
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MhcWeights:
  """Structured layer weights container for mHC-lite."""

  norm_scale: jax.Array
  w_pre: jax.Array
  b_pre: jax.Array
  alpha_pre: jax.Array
  w_post: jax.Array
  b_post: jax.Array
  alpha_post: jax.Array
  w_res: jax.Array
  b_res: jax.Array
  alpha_res: jax.Array

  def fold_norm_scale(self) -> jax.Array:
    return fold_norm_scale(self.norm_scale, self.w_pre, self.w_post, self.w_res)

  def to_coeff_params(self) -> "MhcCoeffParams":
    return MhcCoeffParams(
        phi=self.fold_norm_scale(),
        alpha_pre=self.alpha_pre,
        b_pre=self.b_pre,
        alpha_post=self.alpha_post,
        b_post=self.b_post,
        alpha_res=self.alpha_res,
        b_res=self.b_res,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class HcWeights:
  """Structured layer weights container for static Hyper-Connections (HC)."""

  w_pre: jax.Array
  w_post: jax.Array
  w_res: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MhcCoeffParams:
  """Parameters required by the coefficient kernel."""

  phi: jax.Array
  alpha_pre: jax.Array
  b_pre: jax.Array
  alpha_post: jax.Array
  b_post: jax.Array
  alpha_res: jax.Array
  b_res: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MhcCoeffOutputs:
  """Outputs generated by the coefficient kernel."""

  h_pre: jax.Array
  h_post: jax.Array
  residual: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MhcCoeffGradients:
  """Backward gradients matching MhcCoeffParams."""

  phi: jax.Array
  alpha_pre: jax.Array
  b_pre: jax.Array
  alpha_post: jax.Array
  b_post: jax.Array
  alpha_res: jax.Array
  b_res: jax.Array


def _token_index_map(num_trailing_dims: int, i):
  """Index map tiling the leading token dimension."""
  return (i,) + (0,) * num_trailing_dims


def _feature_tiled_index_map(num_middle_dims: int, token, feature):
  """Index map tiling both leading token and trailing feature dimensions."""
  return (token,) + (0,) * num_middle_dims + (feature,)


def _untiled_feature_index_map(num_trailing_dims: int, token, feature):
  """Index map tiling only the leading token dimension on a 2D grid."""
  del feature
  return (token,) + (0,) * num_trailing_dims


def token_block_spec(shape: tuple[int, ...], block_size: int) -> pl.BlockSpec:
  """Returns a BlockSpec tiling the leading token dimension."""
  block_shape = (block_size,) + shape[1:]
  return pl.BlockSpec(
      block_shape, functools.partial(_token_index_map, len(shape) - 1)
  )


def feature_tiled_block_spec(
    shape: tuple[int, ...],
    block_size: int,
    feature_block_size: int,
    tiled_feature: bool = True,
) -> pl.BlockSpec:
  """Returns a 2D BlockSpec over (token_idx, feature_idx) grid."""
  if tiled_feature:
    block_shape = (block_size,) + shape[1:-1] + (feature_block_size,)
    return pl.BlockSpec(
        block_shape,
        functools.partial(_feature_tiled_index_map, len(shape) - 2),
    )
  block_shape = (block_size,) + shape[1:]
  return pl.BlockSpec(
      block_shape,
      functools.partial(_untiled_feature_index_map, len(shape) - 1),
  )


def fold_norm_scale(norm_scale, w_pre, w_post, w_res) -> jax.Array:
  """Folds the RMSNorm channel scale into the three projection matrices."""
  weights = jnp.concatenate((w_pre, w_post, w_res), axis=-1).astype(jnp.float32)
  return norm_scale.astype(jnp.float32)[:, None] * weights


def validate_token_block_size(
    tokens: int, block_size: int, *, name: str
) -> None:
  """Validates a token-axis Pallas block size."""
  if block_size < 8 or block_size % 8:
    raise UnsupportedInputError(
        f"{name} must be a positive multiple of 8; got {block_size}."
    )
  if tokens % block_size:
    raise UnsupportedInputError(
        f"The per-device token count ({tokens}) must be divisible by {name}"
        f" ({block_size})."
    )


def validate_feature_block_size(embedding: int, block_size: int) -> None:
  """Validates the feature tile used by the post-application backward."""
  if block_size < 128 or block_size % 128:
    raise UnsupportedInputError(
        "bwd_feature_block_size must be a positive multiple of 128; got"
        f" {block_size}."
    )
  if embedding % block_size:
    raise UnsupportedInputError(
        f"The embedding dimension ({embedding}) must be divisible by"
        f" bwd_feature_block_size ({block_size})."
    )


def validate_inputs(
    x: jax.Array,
    block_size: int,
    permutations_shape: tuple[int, ...] | None = None,
    *,
    block_size_name: str = "block_size",
) -> None:
  """Validates the shape, dtype, and forward token block constraints."""
  if x.dtype != jnp.bfloat16:
    raise UnsupportedInputError(
        f"The mHC Pallas kernel requires bfloat16 activations; got {x.dtype}."
    )
  if x.ndim != 4:
    raise UnsupportedInputError(
        "Expected x to have shape (batch, sequence, streams, embedding); got"
        f" {x.shape}."
    )
  batch, sequence, streams, embedding = x.shape
  if streams != 4 or (
      permutations_shape is not None and permutations_shape != (24, 4, 4)
  ):
    raise UnsupportedInputError(
        "The optimized mHC Pallas kernel currently supports mHC-lite with"
        f" expansion rate 4 only; got x.shape={x.shape} and"
        f" permutations.shape={permutations_shape}."
    )
  if embedding % 128:
    raise UnsupportedInputError(
        f"The embedding dimension must be divisible by 128; got {embedding}."
    )
  validate_token_block_size(batch * sequence, block_size, name=block_size_name)
