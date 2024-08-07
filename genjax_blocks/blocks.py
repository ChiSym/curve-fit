import math
from typing import Generator

import genjax

from penzai import pz
import jax.numpy as jnp
import jax.random
from genjax import Pytree, ChoiceMapBuilder as C
from genjax.core import GenerativeFunctionClosure, GenerativeFunction
from genjax.typing import Callable, FloatArray, PRNGKey, ArrayLike, Tuple, List
import genstudio.plot as Plot


class Block:
    """A Block represents a distribution of functions. Blocks can be composed
    with pointwise operations, the result being distributions over expression
    trees of a fixed shape. Blocks may be sampled to generate BlockFunctions
    from the underlying distribution."""

    params_distribution: GenerativeFunction
    function_family: Callable
    gf: GenerativeFunction
    jitted_sample: Callable

    def __init__(self, params_distribution, function_family):
        self.params_distribution = params_distribution
        self.function_family = function_family

        @genjax.gen
        def gf():
            params = params_distribution() @ "curve_params"
            return BlockFunction(params, function_family)
        self.gf = gf
        self.jitted_sample = jax.jit(gf.simulate)

    def sample(self, n: int = 1, k: PRNGKey = jax.random.PRNGKey(0)):
        return jax.vmap(
            self.jitted_sample, in_axes=(0, None)
        )(
            jax.random.split(k, n), ()
        )

    def constraint_from_params(self, params):
        return NotImplementedError()

    def curve_from_params(self, params):
        return self.gf.assess(C["curve_params"].set(self.constraint_from_params(params)), ())[1]

    def __add__(self, b: "Block"):
        return Pointwise(self, b, jnp.add)

    def __mul__(self, b: "Block"):
        return Pointwise(self, b, jnp.multiply)

    def __matmul__(self, b: "Block"):
        return Compose(self, b)

    def address_segments(self) -> Generator[Tuple, None, None]:
        raise NotImplementedError()


@pz.pytree_dataclass
class BlockFunction(Pytree):
    """A BlockFunction is a Pytree which is also Callable."""

    params: FloatArray | Tuple
    function_family: Callable = Pytree.static()

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
            return coef.repeat(n=max_degree+1)() @ "p"

        def function_family(params, x):
            deg = params.shape[-1]
            # tricky: we don't want pow to act like a binary operation between two
            # arrays of the same shape; instead, we want it to take each element
            # of the LHS and raise it to each of the powers in the RHS. So we convert
            # the LHS into an (N, 1) shape.
            powers = jnp.pow(jnp.array(x)[jnp.newaxis].T, jnp.arange(deg))
            return powers @ params.T

        super().__init__(params_distribution, function_family)

    def constraint_from_params(self, params):
        return C["p", jnp.arange(len(params)), "coefficient"].set(params)

    def address_segments(self):
        yield ("p", ..., "coefficient")


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

        super().__init__(params_distribution, function_family)

    def constraint_from_params(self, params):
        amplitude, phase, frequency = params.T
        return C.d({"a": amplitude, "φ": phase, "ω": frequency})

    def address_segments(self):
        yield ("a",)
        yield ("φ",)
        yield ("ω",)


class Exponential(Block):
    def __init__(self, *, a: GenerativeFunctionClosure, b: GenerativeFunctionClosure):
        @genjax.gen
        def params_distribution():
            return jnp.array([a @ "a", b @ "b"])

        def function_family(params, x):
            a, b = params.T
            return a * jnp.exp(b * x)

        super().__init__(params_distribution, function_family)

    def constraint_from_params(self, params):
        a, b = params.T
        return C.d({"a": a, "b": b})

    def address_segments(self):
        yield ("a",)
        yield ("b",)


class Pointwise(Block):
    """Combines two blocks into an expression tree using a supplied concrete
    binary operation. Generates BlockFunctions from the distribution of
    expression trees with this fixed shape."""

    # NB: These are not commutative, even if the underlying binary operation is,
    # due to the way randomness is threaded through the operands.
    def __init__(
        self, f: Block, g: Block, op: Callable[[ArrayLike, ArrayLike], FloatArray]
    ):
        self.f = f
        self.g = g

        @genjax.gen
        def params_distribution():
            return f.params_distribution() @ "l", g.params_distribution() @ "r"

        def function_family(params, x):
            params_f, params_g = params
            return op(f.function_family(params_f, x), g.function_family(params_g, x))

        super().__init__(params_distribution, function_family)

    def constraint_from_params(self, params):
        params_f, params_g = params
        return C.d({
            "l": self.f.constraint_from_params(params_f),
            "r": self.g.constraint_from_params(params_g)
        })

    def address_segments(self):
        for s in self.f.address_segments():
            yield ("l",) + s
        for s in self.g.address_segments():
            yield ("r",) + s


class Compose(Block):
    """Combines two blocks using function compostion. `Compose(f, g)` represents
    `f(g(_))`."""

    def __init__(self, f: Block, g: Block):
        self.f = f
        self.g = g

        @genjax.gen
        def params_distribution():
            return f.params_distribution() @ "l", g.params_distribution() @ "r"

        def function_family(params, x):
            params_f, params_g = params
            return f.function_family(params_f, g.function_family(params_g, x))

        super().__init__(params_distribution, function_family)

    def constraint_from_params(self, params):
        params_f, params_g = params
        return C.d({
            "l": self.f.constraint_from_params(params_f),
            "r": self.g.constraint_from_params(params_g)
        })

    def address_segments(self):
        for s in self.f.address_segments():
            yield ("l",) + s
        for s in self.g.address_segments():
            yield ("r",) + s


class CurveFit:
    """A CurveFit takes a Block, distribution of sigma_inlier and p_outlier,
    and produces an object capable of producing importance samples of the
    function distribution induced by the Block using JAX acceleration."""

    gf: GenerativeFunction
    curve: Block
    jitted_importance: Callable
    coefficient_paths: List[Tuple]
    categorical_sampler: Callable

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
        self.coefficient_paths = [("curve", "params") + p for p in self.curve.address_segments()]
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
            Plot.line({"x": xs, "y": ys},
                      curve="cardinal-open",
                      stroke="black" if winner(i) else i%12,
                      strokeWidth=4 if winner(i) else 1)
            for i, ys in enumerate(yss.T)
        ],
        Plot.domain([-1, 1]),
        {"clip": True, "height": 400, "width": 400},
    )


class DataModel:
    params_distribution: GenerativeFunction
    kernel: GenerativeFunction
    gf: GenerativeFunction
    jitted_sample: Callable

    def __init__(self, params_distribution, kernel):
        self.params_distribution = params_distribution
        self.kernel = kernel

        @genjax.gen
        def gf(ys_latent: ArrayLike) -> FloatArray:
            params = params_distribution() @ "kernel_params"
            return kernel.vmap(in_axes=(0, None))(ys_latent, params) @ "kernel"
        self.gf = gf
        self.jitted_sample = jax.jit(gf.simulate)

    def sample(self, ys_latent: ArrayLike, n: int = 1, k: PRNGKey = jax.random.PRNGKey(0)):
        return jax.vmap(
            self.jitted_sample, in_axes=(0, None)
        )(
            jax.random.split(k, n), (ys_latent,)
        )

    def constraint_from_params(self, params):
        return NotImplementedError()

    def constraint_from_samples(self, samples):
        return NotImplementedError()


class NoisyData(DataModel):
    def __init__(self, *, sigma_inlier: GenerativeFunctionClosure):
        @genjax.gen
        def params_distribution():
            return sigma_inlier @ "σ_inlier"

        @genjax.gen
        def kernel(
            y_latent: ArrayLike,
            params: ArrayLike
        ) -> FloatArray:
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
    def __init__(self, *, sigma_inlier: GenerativeFunctionClosure, p_outlier: GenerativeFunctionClosure):
        @genjax.gen
        def params_distribution():
            sigma_in = sigma_inlier @ "σ_inlier"
            p_out = p_outlier @ "p_outlier"
            return (sigma_in, p_out)

        @genjax.gen
        def kernel(
            y_latent: ArrayLike,
            params: Tuple
        ) -> FloatArray:
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
    jitted_sample: Callable
    jitted_importance: Callable
    coefficient_paths: List[Tuple]

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

        self.coefficient_paths = [("curve", "curve_params") + p for p in curve.address_segments()]

    def sample(self, n: int = 1, k: PRNGKey = jax.random.PRNGKey(0)):
        return jax.vmap(
            self.jitted_sample, in_axes=(0, None)
        )(
            jax.random.split(k, n), ()
        )

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

        constraint = C["data", "kernel", jnp.arange(len(ys)), "y"].set(ys)
        samples, log_weights = jax.vmap(
            self.jitted_importance, in_axes=(0, None, None)
        )(
            jax.random.split(key1, N * K), constraint, (xs,)
        )

        # reshape the samples in to K batches of size N
        log_weights = log_weights.reshape((K, N))
        winners = jax.vmap(categorical_sampler)(
            jax.random.split(key2, K), log_weights
        )
        # indices returned are relative to the start of the batch from which they were drawn.
        # globalize the indices by adding back the index of the start of each batch.
        winners += jnp.arange(0, N * K, N)

        return jax.tree.map(lambda x: x[winners], samples)

    def log_density(self, curve_params, kernel_params, xs, samples):
        constraint = C.d({
            "curve": C.d({"curve_params": self.curve.constraint_from_params(curve_params)}),
            "data": C.d({
                "kernel_params": self.data_model.constraint_from_params(kernel_params),
                "kernel": C[jnp.arange(len(samples))].set(self.data_model.constraint_from_samples(samples))
            })
        })
        return self.gf.assess(constraint, (xs,))[0]
