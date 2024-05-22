import math
from typing import override

import genjax
from penzai import pz
import jax.numpy as jnp
import jax.random
from genjax import Pytree
from genjax.generative_functions.static import StaticGenerativeFunction
from genjax.core import GenerativeFunctionClosure, GenerativeFunction
from genjax.typing import Callable, FloatArray, PRNGKey
from jax.typing import (
    ArrayLike,
)  # TODO: this appears in GenJAX 0.4.0; update this when that's released


class Block:
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
        return Pointwise(self, b, lambda a, b: jnp.add(a, b))

    def __mul__(self, b: "Block"):
        return Pointwise(self, b, lambda a, b: jnp.multiply(a, b))

    def __matmul__(self, b: "Block"):
        return Compose(self, b)


class BlockFunction(Pytree):
    """A BlockFunction is a Pytree which is also Callable."""

    def __call__(self, x: ArrayLike) -> FloatArray:
        raise NotImplementedError


@pz.pytree_dataclass
class BinaryOperation(BlockFunction):
    lhs: BlockFunction
    rhs: BlockFunction
    op: Callable[[ArrayLike, ArrayLike], FloatArray] = Pytree.static()

    def __call__(self, x: ArrayLike) -> FloatArray:
        return self.op(self.lhs(x), self.rhs(x))


class Polynomial(Block):
    def __init__(self, *, max_degree: int, coefficient_d: GenerativeFunctionClosure):
        @genjax.combinators.repeat_combinator(num_repeats=max_degree + 1)
        @genjax.gen
        def coefficient_gf() -> FloatArray:
            return coefficient_d @ "coefficients"

        @genjax.gen
        def polynomial_gf() -> BlockFunction:
            return Polynomial.Function(coefficient_gf() @ "p")

        super().__init__(polynomial_gf)

    @pz.pytree_dataclass
    class Function(BlockFunction):
        coefficients: FloatArray

        @override
        def __call__(self, x: ArrayLike):
            deg = self.coefficients.shape[-1]
            powers = jnp.pow(
                jnp.broadcast_to(x, deg), jax.lax.iota(dtype=int, size=deg)
            )
            return jax.numpy.matmul(self.coefficients, powers)


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

    @pz.pytree_dataclass
    class Function(BlockFunction):
        amplitude: FloatArray
        phase: FloatArray
        period: FloatArray

        @override
        def __call__(self, x: ArrayLike) -> FloatArray:
            return self.amplitude * jnp.sin(self.phase + 2 * x * math.pi / self.period)


class Exponential(Block):
    def __init__(self, *, a: GenerativeFunctionClosure, b: GenerativeFunctionClosure):
        @genjax.gen
        def exponential_gf() -> BlockFunction:
            return Exponential.Function(a @ "a", b @ "b")

        super().__init__(exponential_gf)

    @pz.pytree_dataclass
    class Function(BlockFunction):
        a: FloatArray
        b: FloatArray

        @override
        def __call__(self, x: ArrayLike) -> FloatArray:
            return self.a * jnp.exp(self.b * x)


class Pointwise(Block):
    # NB: These are not commutative, even if the underlying binary operation is,
    # due to the way randomness is threaded through the operands.
    def __init__(
        self, f: Block, g: Block, op: Callable[[ArrayLike, ArrayLike], FloatArray]
    ):
        self.f = f
        self.g = g

        @genjax.gen
        def pointwise_op() -> BlockFunction:
            return BinaryOperation(f.gf() @ "l", g.gf() @ "r", op)

        super().__init__(pointwise_op)


class Compose(Block):
    def __init__(self, f: Block, g: Block):
        @genjax.gen
        def composition() -> BlockFunction:
            return Compose.Function(f.gf() @ "l", g.gf() @ "r")

        super().__init__(composition)

    @pz.pytree_dataclass
    class Function(BlockFunction):
        f: BlockFunction
        g: BlockFunction

        @override
        def __call__(self, x: ArrayLike) -> FloatArray:
            return self.f(self.g(x))


class CoinToss(Block):
    def __init__(self, probability: float, heads: Block, tails: Block):
        swc = genjax.switch_combinator(tails.gf, heads.gf)

        @genjax.gen
        def coin_toss_gf() -> StaticGenerativeFunction:
            a = jnp.array(genjax.flip(probability) @ "coin", dtype=int)
            choice = swc(a, (), ()) @ "toss"
            return choice

        super().__init__(coin_toss_gf)


class CurveFit:
    gf: GenerativeFunction
    jitted_importance: Callable

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

        swc = genjax.switch_combinator(inlier_model, outlier_model)

        @genjax.combinators.vmap_combinator(in_axes=(0, None, None, None))
        @genjax.gen
        def kernel(
            x: ArrayLike,
            f: Callable[[ArrayLike], FloatArray],
            sigma_in: ArrayLike,
            p_out: ArrayLike,
        ) -> StaticGenerativeFunction:
            is_outlier = genjax.flip(p_out) @ "outlier"
            io = jnp.array(is_outlier, dtype=int)
            return swc(io, (f(x), sigma_in), ()) @ "y"

        @genjax.gen
        def model(xs: FloatArray) -> FloatArray:
            c = curve.gf() @ "curve"
            sigma_in = sigma_inlier @ "sigma_inlier"
            p_out = p_outlier @ "p_outlier"
            _ = kernel(xs, c, sigma_in, p_out) @ "ys"
            return c

        self.gf = model
        self.jitted_importance = jax.jit(self.gf.importance)

    def importance_sample(
        self,
        xs: FloatArray,
        ys: FloatArray,
        N: int,
        K: int,
        key: PRNGKey = jax.random.PRNGKey(0),
    ):
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

        # curves = trs.get_subtrace("curve").get_retval()
        # return jax.tree.map(lambda x: x[ixs], curves)
        selected = jax.tree.map(lambda x: x[ixs], trs)
        return selected
