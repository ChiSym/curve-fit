import math
from typing import Generator

import genjax

from penzai import pz
import jax.numpy as jnp
import jax.random
from genjax import Pytree
from genjax.core import GenerativeFunctionClosure, GenerativeFunction
from genjax.typing import Callable, FloatArray, PRNGKey, ArrayLike, Tuple, List
import genstudio.plot as Plot


class Block:
    """A Block represents a distribution of functions. Blocks can be composed
    with pointwise operations, the result being distributions over expression
    trees of a fixed shape. Blocks may be sampled to generate BlockFunctions
    from the underlying distribution."""

    gf: GenerativeFunction
    jitted_sample: Callable

    def __init__(self, gf):
        self.gf = gf
        self.jitted_sample = jax.jit(self.gf.simulate)

    def sample(self, n: int = 1, k: PRNGKey = jax.random.PRNGKey(0)):
        return jax.vmap(self.jitted_sample, in_axes=(0, None))(
            jax.random.split(k, n), ()
        )

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

    params: FloatArray | tuple[Block]
    function_family: Callable = Pytree.static()

    def __call__(self, x: ArrayLike) -> FloatArray:
        return self.function_family(self.params, x)

    def params_grad(self, x: ArrayLike) -> FloatArray:
        return jax.jacfwd(self.function_family)(self.params, x)


class Polynomial(Block):
    def __init__(self, *, max_degree: int, coefficient_d: GenerativeFunctionClosure):
        def polynomial(params, x):
            deg = params.shape[-1]
            # tricky: we don't want pow to act like a binary operation between two
            # arrays of the same shape; instead, we want it to take each element
            # of the LHS and raise it to each of the powers in the RHS. So we convert
            # the LHS into an (N, 1) shape.
            powers = jnp.pow(jnp.array(x)[jnp.newaxis].T, jnp.arange(deg))
            return powers @ params.T

        # This trampoline from GenerativeFunctionClosure to GenerativeFunction is required
        # because of GEN-420 (or so it is conjectured)
        @genjax.gen
        def coef():
            return coefficient_d @ "coefficients"

        @genjax.gen
        def polynomial_gf() -> BlockFunction:
            params = coef.repeat(n=max_degree+1)() @ "p"
            return BlockFunction(params, polynomial)

        super().__init__(polynomial_gf)

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
        def periodic(params, x):
            amplitude, phase, frequency = params.T
            return amplitude * jnp.sin(2 * math.pi * frequency * (x + phase))

        @genjax.gen
        def periodic_gf() -> BlockFunction:
            params = jnp.array([amplitude @ "a", phase @ "φ", frequency @ "ω"])
            return BlockFunction(params, periodic)

        super().__init__(periodic_gf)

    def address_segments(self):
        yield ("a",)
        yield ("φ",)
        yield ("ω",)


class Exponential(Block):

    def __init__(self, *, a: GenerativeFunctionClosure, b: GenerativeFunctionClosure):
        def exponential(params, x):
            a, b = params.T
            return a * jnp.exp(b * x)

        @genjax.gen
        def exponential_gf() -> BlockFunction:
            params = jnp.array([a @ "a", b @ "b"])
            return BlockFunction(params, exponential)

        super().__init__(exponential_gf)

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
        self.op = op

        def pointwise(params, x):
            block_f, block_g = params
            return self.op(block_f(x), block_g(x))

        @genjax.gen
        def pointwise_gf() -> BlockFunction:
            params = self.f.gf() @ "l", self.g.gf() @ "r"
            return BlockFunction(params, pointwise)

        super().__init__(pointwise_gf)

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

        def composition(params, x):
            block_f, block_g = params
            return block_f(block_g(x))

        @genjax.gen
        def composite_gf() -> BlockFunction:
            params = self.f.gf() @ "l", self.g.gf() @ "r"
            return BlockFunction(params, composition)

        super().__init__(composite_gf)

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
        self.coefficient_paths = [("curve",) + p for p in self.curve.address_segments()]
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
