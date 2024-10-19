import math
from typing import Generator

import genjax

from penzai import pz
import jax.numpy as jnp
import jax.random
from genjax import Pytree, ChoiceMapBuilder as C
from genjax.core import GenerativeFunctionClosure, GenerativeFunction
from genjax.typing import Callable, FloatArray, PRNGKey, ArrayLike

import genstudio.plot as Plot


class Block:
    """A Block represents a distribution of functions. Blocks can be composed
    with pointwise operations, the result being distributions over expression
    trees of a fixed shape. Blocks may be sampled to generate BlockFunctions
    from the underlying distribution."""

    params_distribution: GenerativeFunction
    function_family: Callable  # type: ignore
    gf: GenerativeFunction
    jitted_sample: Callable  # type: ignore
    address_segments: list[tuple]

    def __init__(self, params_distribution, function_family, address_segments):
        self.params_distribution = params_distribution
        self.function_family = function_family
        self.address_segments = address_segments

        @genjax.gen
        def gf():
            params = params_distribution() @ "curve_params"
            return BlockFunction(params, function_family)

        self.gf = gf
        self.jitted_sample = jax.jit(gf.simulate)

    def sample(self, n: int = 1, k: PRNGKey = jax.random.PRNGKey(0)):
        return jax.vmap(self.jitted_sample, in_axes=(0, None))(
            jax.random.split(k, n), ()
        )

    def constraint_from_params(self, params):
        return C.d({"NotImplemented": params})

    def curve_from_params(self, params):
        return BlockFunction(params, self.function_family)

    def __add__(self, b: "Block"):
        return Pointwise(self, b, jnp.add)

    def __mul__(self, b: "Block"):
        return Pointwise(self, b, jnp.multiply)

    def __matmul__(self, b: "Block"):
        return Compose(self, b)


@pz.pytree_dataclass
class BlockFunction(Pytree):
    """A BlockFunction is a Pytree which is also Callable."""

    params: FloatArray | tuple
    function_family: Callable = Pytree.static()  # type: ignore

    def __call__(self, x: ArrayLike) -> FloatArray:
        return self.function_family(self.params, x)

    def params_grad(self, x: ArrayLike) -> FloatArray:
        return jax.jacfwd(self.function_family)(self.params, x)


class Polynomial(Block):
    def __init__(self, *, max_degree: int, coefficient_d: GenerativeFunctionClosure):
        # This trampoline from GenerativeFunctionClosure to GenerativeFunction is required
        # because of GEN-420 (or so it is conjectured)
        @genjax.gen
        def coef():
            return coefficient_d @ "coefficient"

        @genjax.gen
        def params_distribution():
            return coef.repeat(n=max_degree + 1)() @ "p"

        def function_family(params, x):
            deg = params.shape[-1]
            # tricky: we don't want pow to act like a binary operation between two
            # arrays of the same shape; instead, we want it to take each element
            # of the LHS and raise it to each of the powers in the RHS. So we convert
            # the LHS into an (N, 1) shape.
            powers = jnp.pow(jnp.array(x)[jnp.newaxis].T, jnp.arange(deg))
            return powers @ params.T

        address_segments = [("p", ..., "coefficient")]

        super().__init__(params_distribution, function_family, address_segments)

    def constraint_from_params(self, params):
        return C["p", jnp.arange(len(params)), "coefficient"].set(params)


class Periodic(Block):
    def __init__(
        self,
        *,
        amplitude: GenerativeFunctionClosure,
        phase: GenerativeFunctionClosure,
        frequency: GenerativeFunctionClosure,
    ):
        @genjax.gen
        def params_distribution():
            return jnp.array([amplitude @ "a", phase @ "φ", frequency @ "ω"])

        def function_family(params, x):
            amplitude, phase, frequency = params.T
            return amplitude * jnp.sin(2 * math.pi * frequency * (x + phase))

        address_segments = [("a",), ("φ",), ("ω",)]

        super().__init__(params_distribution, function_family, address_segments)

    def constraint_from_params(self, params):
        amplitude, phase, frequency = params.T
        return C.d({"a": amplitude, "φ": phase, "ω": frequency})


class Exponential(Block):
    def __init__(self, *, a: GenerativeFunctionClosure, b: GenerativeFunctionClosure):
        @genjax.gen
        def params_distribution():
            return jnp.array([a @ "a", b @ "b"])

        def function_family(params, x):
            a, b = params.T
            return a * jnp.exp(b * x)

        address_segments = [("a",), ("b",)]

        super().__init__(params_distribution, function_family, address_segments)

    def constraint_from_params(self, params):
        a, b = params.T
        return C.d({"a": a, "b": b})


class Pointwise(Block):
    """Combines two blocks into an expression tree using a supplied concrete
    binary operation. Generates BlockFunctions from the distribution of
    expression trees with this fixed shape."""

    # NB: These are not commutative, even if the underlying binary operation is,
    # due to the way randomness is threaded through the operands.
    def __init__(
        self, l: Block, r: Block, op: Callable[[ArrayLike, ArrayLike], FloatArray]
    ):
        self.l = l
        self.r = r

        @genjax.gen
        def params_distribution():
            return l.params_distribution() @ "l", r.params_distribution() @ "r"

        def function_family(params, x):
            params_l, params_r = params
            return op(l.function_family(params_l, x), r.function_family(params_r, x))

        address_segments = [("l",) + s for s in self.l.address_segments] + [
            ("r",) + s for s in self.l.address_segments
        ]

        super().__init__(params_distribution, function_family, address_segments)

    def constraint_from_params(self, params):
        params_l, params_r = params
        return C.d(
            {
                "l": self.l.constraint_from_params(params_l),
                "r": self.r.constraint_from_params(params_r),
            }
        )


class Compose(Block):
    """Combines two blocks using function compostion. `Compose(f, g)` represents
    `f(g(_))`."""

    def __init__(self, l: Block, r: Block):
        self.l = l
        self.r = r

        @genjax.gen
        def params_distribution():
            return l.params_distribution() @ "l", r.params_distribution() @ "r"

        def function_family(params, x):
            params_l, params_r = params
            return l.function_family(params_l, r.function_family(params_r, x))

        address_segments = [("l",) + s for s in self.l.address_segments] + [
            ("r",) + s for s in self.l.address_segments
        ]

        super().__init__(params_distribution, function_family, address_segments)

    def constraint_from_params(self, params):
        params_l, params_r = params
        return C.d(
            {
                "l": self.l.constraint_from_params(params_l),
                "r": self.r.constraint_from_params(params_r),
            }
        )


class CurveFit:
    """A CurveFit takes a Block, distribution of sigma_inlier and p_outlier,
    and produces an object capable of producing importance samples of the
    function distribution induced by the Block using JAX acceleration."""

    gf: GenerativeFunction
    curve: Block
    jitted_importance: Callable  # type: ignore
    coefficient_paths: list[tuple]
    categorical_sampler: Callable  # type: ignore

    def __init__(
        self,
        *,
        curve: Block,
        sigma_inlier: GenerativeFunctionClosure,
        p_outlier: GenerativeFunctionClosure,
    ):
        @genjax.gen
        def inlier_model(y, sigma_in):
            return genjax.normal(y, sigma_in) @ "value"

        @genjax.gen
        def outlier_model():
            return genjax.uniform(-1.0, 1.0) @ "value"

        fork = outlier_model.or_else(inlier_model)

        @genjax.combinators.vmap(in_axes=(0, None, None, None))
        @genjax.gen
        def kernel(
            x: ArrayLike,
            f: Callable[[ArrayLike], FloatArray],
            sigma_in: ArrayLike,
            p_out: ArrayLike,
        ) -> FloatArray:
            is_outlier = genjax.flip(p_out) @ "outlier"
            return fork(jnp.bool_(is_outlier), (), (f(x), sigma_in)) @ "y"

        @genjax.gen
        def model(xs: FloatArray) -> FloatArray:
            c = curve.gf() @ "curve"
            sigma_in = sigma_inlier @ "sigma_inlier"
            p_out = p_outlier @ "p_outlier"
            return kernel(xs, c, sigma_in, p_out) @ "ys"

        self.gf = model
        self.curve = curve
        self.jitted_importance = jax.jit(self.gf.importance)
        self.coefficient_paths = [
            ("curve", "curve_params") + p for p in self.curve.address_segments
        ]
        self.categorical_sampler = jax.jit(genjax.categorical.sampler)

    def importance_sample(
        self,
        xs: FloatArray,
        ys: FloatArray,
        N: int,
        K: int,
        key: PRNGKey = jax.random.PRNGKey(0),
    ):
        """Generate $K$ importance samples of the curves fitted to $xs, ys$.
        Each sample will be drawn from a separate weighted categorical distribution
        of $N$ importance samples (so that $NK$ samples are taken overall)."""
        choose_ys = jax.vmap(
            lambda ix, v: genjax.ChoiceMapBuilder["ys", ix, "y", "value"].set(v),
        )(jnp.arange(len(ys)), ys)

        key1, key2 = jax.random.split(key)
        samples, log_weights = jax.vmap(
            self.jitted_importance, in_axes=(0, None, None)
        )(jax.random.split(key1, N * K), choose_ys, (xs,))
        # reshape the samples in to K batches of size N
        log_weights = log_weights.reshape((K, N))
        winners = jax.vmap(self.categorical_sampler)(
            jax.random.split(key2, K), log_weights
        )
        # indices returned are relative to the start of the batch from which they were drawn.
        # globalize the indices by adding back the index of the start of each batch.
        winners += jnp.arange(0, N * K, N)
        return jax.tree.map(lambda x: x[winners], samples)


def plot_functions(fns: BlockFunction, winningIndex=None, **kwargs):
    xs = jnp.linspace(-1, 1, 40)
    yss = jax.vmap(fns)(xs)

    def winner(i):
        return i == winningIndex

    return Plot.new(
        [
            Plot.line(
                {"x": xs, "y": ys},
                curve="cardinal-open",
                stroke="black" if winner(i) else i % 12,
                strokeWidth=4 if winner(i) else 1,
            )
            for i, ys in enumerate(yss.T)
        ],
        Plot.domain([-1, 1]),
        {"clip": True, "height": 400, "width": 400},
    )


class DataModel:
    params_distribution: GenerativeFunction
    kernel: GenerativeFunction
    gf: GenerativeFunction
    jitted_sample: Callable  # type: ignore

    def __init__(self, params_distribution, kernel):
        self.params_distribution = params_distribution
        self.kernel = kernel

        @genjax.gen
        def gf(ys_latent: ArrayLike) -> FloatArray:
            params = params_distribution() @ "kernel_params"
            return kernel.vmap(in_axes=(0, None))(ys_latent, params) @ "kernel"

        self.gf = gf
        self.jitted_sample = jax.jit(gf.simulate)

    def sample(
        self, ys_latent: ArrayLike, n: int = 1, k: PRNGKey = jax.random.PRNGKey(0)
    ):
        return jax.vmap(self.jitted_sample, in_axes=(0, None))(
            jax.random.split(k, n), (ys_latent,)
        )

    def constraint_from_params(self, params):
        return C.d({"NotImplemented": params})

    def constraint_from_samples(self, samples):
        return C.d({"NotImplemented": samples})


class NoisyData(DataModel):
    def __init__(self, *, sigma_inlier: GenerativeFunctionClosure):
        @genjax.gen
        def params_distribution():
            return sigma_inlier @ "σ_inlier"

        @genjax.gen
        def kernel(y_latent: ArrayLike, params: ArrayLike) -> FloatArray:
            sigma_in = params
            return genjax.normal(y_latent, sigma_in) @ "y"

        super().__init__(params_distribution, kernel)

    def constraint_from_params(self, params):
        sigma_in = params
        return C.d({"σ_inlier": sigma_in})

    def constraint_from_samples(self, samples):
        y = samples
        return C.d({"y": y})


class NoisyOutliersData(DataModel):
    def __init__(
        self,
        *,
        sigma_inlier: GenerativeFunctionClosure,
        p_outlier: GenerativeFunctionClosure,
    ):
        @genjax.gen
        def params_distribution():
            sigma_in = sigma_inlier @ "σ_inlier"
            p_out = p_outlier @ "p_outlier"
            return (sigma_in, p_out)

        @genjax.gen
        def kernel(y_latent: ArrayLike, params: tuple) -> FloatArray:
            sigma_in, p_out = params

            inlier_model = genjax.normal(y_latent, sigma_in)
            outlier_model = genjax.uniform(-1.0, 1.0)
            branch_model = outlier_model.or_else(inlier_model)

            is_outlier = jnp.bool_(genjax.flip(p_out) @ "outlier")
            return branch_model(is_outlier, (), ()) @ "y"

        super().__init__(params_distribution, kernel)

    def constraint_from_params(self, params):
        sigma_in, p_out = params
        return C.d({"σ_inlier": sigma_in, "p_outlier": p_out})

    def constraint_from_samples(self, samples):
        outlier, y = samples
        return C.d({"outlier": outlier, "y": y})


categorical_sampler = jax.jit(genjax.categorical.sampler)


class CurveDataModel:
    curve: Block
    data_model: DataModel
    gf: GenerativeFunction
    jitted_sample: Callable  # type: ignore
    jitted_importance: Callable  # type: ignore
    coefficient_paths: list[tuple]

    def __init__(self, curve, data_model):
        self.curve = curve
        self.data_model = data_model

        @genjax.gen
        def gf(xs):
            c = curve.gf() @ "curve"
            return data_model.gf(c(xs)) @ "data"

        self.gf = gf
        self.jitted_sample = jax.jit(gf.simulate)
        self.jitted_importance = jax.jit(gf.importance)

        self.coefficient_paths = [
            ("curve", "curve_params") + p for p in curve.address_segments
        ]

    def sample(self, n: int = 1, k: PRNGKey = jax.random.PRNGKey(0)):
        return jax.vmap(self.jitted_sample, in_axes=(0, None))(
            jax.random.split(k, n), ()
        )

    def gradient_ascent_model_params(
        self, params_guess, kernel_params, xs, ys, N_steps=1000, learning_rate=1e-5
    ):
        jitted_grad = jax.jit(
            jax.jacfwd(lambda params: self.log_density(params, kernel_params, xs, ys))
        )

        params_optimized = params_guess
        for _ in range(N_steps):
            params_optimized += learning_rate * jitted_grad(params_optimized)

        return self.curve.curve_from_params(params_optimized)

    def log_density(self, curve_params, kernel_params, xs, samples):
        constraint = C.d(
            {
                "curve": C.d(
                    {"curve_params": self.curve.constraint_from_params(curve_params)}
                ),
                "data": C.d(
                    {
                        "kernel_params": self.data_model.constraint_from_params(
                            kernel_params
                        ),
                        "kernel": C[jnp.arange(len(xs))].set(
                            self.data_model.constraint_from_samples(samples)
                        ),
                    }
                ),
            }
        )
        return self.gf.assess(constraint, (xs,))[0]

    def importance_resample(
        self,
        xs: FloatArray,
        ys: FloatArray,
        N: int,
        K: int,
        key: PRNGKey = jax.random.PRNGKey(0),
    ):
        """Generate $K$ importance samples from the curves fitted to $xs, ys$.
        Each sample will be drawn from a separate weighted categorical distribution
        of $N$ importance samples (so that $NK$ samples are taken overall)."""
        key1, key2 = jax.random.split(key)

        constraint = C.d(
            {
                "data": C.d(
                    {
                        "kernel": C[jnp.arange(len(xs))].set(
                            self.data_model.constraint_from_samples(ys)
                        ),
                    }
                ),
            }
        )
        samples, log_weights = jax.vmap(
            self.jitted_importance, in_axes=(0, None, None)
        )(jax.random.split(key1, N * K), constraint, (xs,))

        # reshape the samples in to K batches of size N
        log_weights = log_weights.reshape((K, N))
        winners = jax.vmap(categorical_sampler)(jax.random.split(key2, K), log_weights)
        # indices returned are relative to the start of the batch from which they were drawn.
        # globalize the indices by adding back the index of the start of each batch.
        winners += jnp.arange(0, N * K, N)

        return jax.tree.map(lambda x: x[winners], samples)
