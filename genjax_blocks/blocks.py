import math
from typing import Generator

import genjax

from penzai import pz
import jax.numpy as jnp
import jax.random
from genjax import Pytree
from genjax.generative_functions.static import StaticGenerativeFunction
from genjax.core import GenerativeFunctionClosure, GenerativeFunction
from genjax.typing import Callable, FloatArray, PRNGKey, ArrayLike, Tuple, List


class Block:
    """A Block represents a distribution of functions. Blocks can be composed
    with pointwise operations, the result being distributions over expression
    trees of a fixed shape. Blocks may be sampled to generate BlockFunctions
    from the underlying distribution."""

    gf: StaticGenerativeFunction
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


class BlockFunction(Pytree):
    """A BlockFunction is a Pytree which is also Callable."""

    def __call__(self, x: ArrayLike) -> FloatArray:
        raise NotImplementedError


class Polynomial(Block):
    def __init__(self, *, max_degree: int, coefficient_d: GenerativeFunctionClosure):
        @genjax.combinators.repeat(n=max_degree + 1)
        @genjax.gen
        def coefficient_gf() -> FloatArray:
            return coefficient_d @ "coefficients"

        @genjax.gen
        def polynomial_gf() -> BlockFunction:
            return Polynomial.Function(coefficient_gf() @ "p")

        super().__init__(polynomial_gf)

    def address_segments(self):
        yield ("p", ..., "coefficients")

    @pz.pytree_dataclass
    class Function(BlockFunction):
        coefficients: FloatArray

        def __call__(self, x: ArrayLike):
            deg = self.coefficients.shape[-1]
            # tricky: we don't want pow to act like a binary operation between two
            # arrays of the same shape; instead, we want it to take each element
            # of the LHS and raise it to each of the powers in the RHS. So we convert
            # the LHS into an (N, 1) shape.
            powers = jnp.pow(jnp.array(x)[jnp.newaxis].T, jnp.arange(deg))
            return powers @ self.coefficients.T


class Periodic(Block):
    def __init__(
        self,
        *,
        amplitude: GenerativeFunctionClosure,
        phase: GenerativeFunctionClosure,
        period: GenerativeFunctionClosure,
    ):
        @genjax.gen
        def periodic_gf() -> BlockFunction:
            return Periodic.Function(amplitude @ "a", phase @ "φ", period @ "T")

        super().__init__(periodic_gf)

    def address_segments(self):
        yield ("T",)
        yield ("a",)
        yield ("φ",)

    @pz.pytree_dataclass
    class Function(BlockFunction):
        amplitude: FloatArray
        phase: FloatArray
        period: FloatArray

        def __call__(self, x: ArrayLike) -> FloatArray:
            return self.amplitude * jnp.sin(self.phase + 2 * x * math.pi / self.period)


class Exponential(Block):
    def __init__(self, *, a: GenerativeFunctionClosure, b: GenerativeFunctionClosure):
        @genjax.gen
        def exponential_gf() -> BlockFunction:
            return Exponential.Function(a @ "a", b @ "b")

        super().__init__(exponential_gf)

    def address_segments(self):
        yield ("a",)
        yield ("b",)

    @pz.pytree_dataclass
    class Function(BlockFunction):
        a: FloatArray
        b: FloatArray

        def __call__(self, x: ArrayLike) -> FloatArray:
            return self.a * jnp.exp(self.b * x)


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
        def pointwise_op() -> BlockFunction:
            return Pointwise.BinaryOperation(f.gf() @ "l", g.gf() @ "r", op)

        super().__init__(pointwise_op)

    def address_segments(self):
        for s in self.f.address_segments():
            yield ("l",) + s
        for s in self.g.address_segments():
            yield ("r",) + s

    @pz.pytree_dataclass
    class BinaryOperation(BlockFunction):
        lhs: BlockFunction
        rhs: BlockFunction
        op: Callable[[ArrayLike, ArrayLike], FloatArray] = Pytree.static()

        def __call__(self, x: ArrayLike) -> FloatArray:
            return self.op(self.lhs(x), self.rhs(x))


class Compose(Block):
    """Combines two blocks using function compostion. `Compose(f, g)` represents
    `f(g(_))`."""

    def __init__(self, f: Block, g: Block):
        self.f = f
        self.g = g

        @genjax.gen
        def composition() -> BlockFunction:
            return Compose.Function(f.gf() @ "l", g.gf() @ "r")

        super().__init__(composition)

    def address_segments(self):
        for s in self.f.address_segments():
            yield ("l",) + s
        for s in self.g.address_segments():
            yield ("r",) + s

    @pz.pytree_dataclass
    class Function(BlockFunction):
        f: BlockFunction
        g: BlockFunction

        def __call__(self, x: ArrayLike) -> FloatArray:
            return self.f(self.g(x))


class CurveFit:
    """A CurveFit takes a Block, distribution of sigma_inlier and p_outlier,
    and produces an object capable of producing importance samples of the
    function distribution induced by the Block using JAX acceleration."""

    gf: GenerativeFunction
    curve: Block
    jitted_importance: Callable
    coefficient_paths: List[Tuple]

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
        ) -> StaticGenerativeFunction:
            is_outlier = genjax.flip(p_out) @ "outlier"
            return fork(jnp.bool_(is_outlier), (), (f(x), sigma_in)) @ "y"

        @genjax.gen
        def model(xs: FloatArray) -> FloatArray:
            c = curve.gf() @ "curve"
            sigma_in = sigma_inlier @ "sigma_inlier"
            p_out = p_outlier @ "p_outlier"
            _ = kernel(xs, c, sigma_in, p_out) @ "ys"
            return c

        self.gf = model
        self.curve = curve
        self.jitted_importance = jax.jit(self.gf.importance)
        self.coefficient_paths = [("curve",) + p for p in self.curve.address_segments()]

    def importance_sample(
        self,
        xs: FloatArray,
        ys: FloatArray,
        N: int,
        K: int,
        key: PRNGKey = jax.random.PRNGKey(0),
    ):
        """Generate $N$ importance samples of the curves fitted to $xs, ys$ and
        sample $K$ of them from the weighted posterior distribution."""
        choose_ys = jax.vmap(
            lambda ix, v: genjax.ChoiceMapBuilder["ys", ix, "y", "value"].set(v),
        )(jnp.arange(len(ys)), ys)

        k1, k2 = jax.random.split(key)
        trs, ws = jax.vmap(self.jitted_importance, in_axes=(0, None, None))(
            jax.random.split(k1, N), choose_ys, (xs,)
        )
        ixs = jax.vmap(jax.jit(genjax.categorical.sampler), in_axes=(0, None))(
            jax.random.split(k2, K), ws
        )
        selected = jax.tree.map(lambda x: x[ixs], trs)
        return selected
