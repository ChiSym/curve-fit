# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: genjax-RWVhKTPb-py3.12
#     language: python
#     name: python3
# ---
# %%
import sys
if "google.colab" in sys.modules:
    from google.colab import auth  # pyright: ignore [reportMissingImports]

    auth.authenticate_user()
    %pip install --quiet keyring keyrings.google-artifactregistry-auth   # type: ignore # noqa
    %pip install --quiet genjax==0.4.0.post4.dev0+9d775c6f genstudio==v2024.06.20.1130 genjax-blocks==0.1.0 --extra-index-url https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/  # noqa # type: ignore
    # This example will work on GPU, CPU or TPU. To change your runtime,
    # select "Change runtime type" from the dropdown on the top right
    # of the colab page.
    #
    # Make sure that the string in brackets below is either `cuda12` (for GPU), `cpu` or `tpu`:
    %pip install --quiet jax[cpu]==0.4.28  # type: ignore
# %% [markdown]
# # DSL for curve fit inference
#
# As part of the `genjax.interpreted` performance investigation, I wanted to investigate a DSL for the curve-fitting task which could achieve JAX-accelerated performance without introducing JAX concepts such as the Switch and Map combinators, `in_axes`, and other things that might complicate the exposition of inference for the newcomer. While doing so, I also studied ways to "automatically" thread randomness through the computation without having to discuss the careful use of `jax.random.split` which we recommend to GenJAX users. Having done the experiment, I have mixed feelings about the results: on the one hand, it is possible to get JAX-level performance with curve-building combinators, but the price of so doing is that the GFI is hidden from view as well, and that may be too far a step. Nonetheless, if you're still interested in what can be achieved in this framework, read on!
# %%
import genjax
from genjax import ChoiceMapBuilder as C
from genjax.typing import PRNGKey, FloatArray, ArrayLike
import genjax_blocks as b
import genstudio.plot as Plot
import jax
import jax.numpy as jnp

genjax.pretty()

# %% [markdown]
# The `blocks` library is concentrated on a simple $\mathbb{R}\rightarrow\mathbb{R}$ inference problem, localized to the unit square for convenience, as found in the introductory GenJAX notebooks. We provide a "basis" of polynomial, $ae^{bx}$, and $a\sin(\phi + 2\pi x/T)$ functions, each of whose parameters are drawn from a standard distribution that the user supplies. The intro curve fit task contemplates a polynomial of degree 2 with normally distributed coefficients, which we may write as

# %%
quadratic = b.Polynomial(max_degree=2, coefficient_d=genjax.normal(0.0, 1.0))

# %% [markdown]
# Note one felicity we have achieved already: we can specify the normal distribution in a simple fashion without having to tuple the arguments or provide randomness: that comes later. The type of `P` is `Block`, a name inspired by [blockly](https://developers.google.com/blockly). The only operation we can do on a Block is request an array of samples from it.

# %%
p = quadratic.sample()
p.get_choices()

# %% [markdown]
# The trace contains choices and a return value. In the blocks DSL, the return value is a JAX-compatible native Python function representing the sampled polynomial $x\mapsto p(x)$. Calling it with the value 0.0 will reveal the constant term:

# %%
p.get_retval()(0.0)

# %% [markdown]
# You may have noticed an extra layer of brackets around the polynomial's coefficients and its return value. This is the magic of JAX: sampling from a block produces an "array BlockFunction". Such an array of functions, when presented with an argument, will return an array of values (one for each function evaluated at that single argument.)
#
# Here's a simple function that will plot a bundle of samples from a block. In this case, we will want to provide a vector of arguments to each function (points along the $x$ axis). Since BlockFunctions are JAX-compatible, we can use `jax.vmap` for that. Now, the functions which already knew how to act as a vector given one argument can act that way across a vector of arguments, producing a grid of points, all in one step.


# %%
def plot_functions(fns: b.BlockFunction, **kwargs):
    xs = jnp.linspace(-1, 1, 200)
    yss = jax.vmap(fns)(xs)
    return Plot.new(
        [
            Plot.line({"x": xs, "y": ys, "stroke": i % 12}, kwargs)
            for i, ys in enumerate(yss.T)
        ],
        {"clip": True, "height": 400, "width": 400, "y": {"domain": [-1, 1]}},
    )


def plot_priors(B: b.Block, n: int):
    return plot_functions(B.sample(n).get_retval())


plot_priors(quadratic, 100)

# %% [markdown]
# We can do the same for our Periodic and Exponential distributions:

# %%
periodic = b.Periodic(
    amplitude=genjax.beta(2.0, 5.0),
    phase=genjax.uniform(-1.0, 1.0),
    period=genjax.normal(1.0, 1.0),
)

exponential = b.Exponential(a=genjax.normal(0.0, 1.0), b=genjax.normal(0.0, 1.0))

# %%
plot_priors(periodic, 50)

# %%
plot_priors(exponential, 50)

# %% [markdown]
# `Block` objects support the pointwise operations $+, *$ as well as `@` (function composition). Vikash's favorite ascending-periodic function might be modeled by the sum of a polynomial and a periodic function which we can write simply as $P+Q$:

# %%
plot_priors(quadratic + periodic, 15)

# %% [markdown]
# It does seem like the goal function lies in the span of the prior distribution in this case. (I pause here to note that pointwise binary operations in `Block` are not commutative as you might expect, because the randomness supplied by `sample` is injected left to right).
#
# The binary operations produce traces representing the expression tree that created them:

# %%
(quadratic + periodic).sample().get_choices()

# %% [markdown]
# Having assembled these pieces, let's turn to the inference task. This begins with a set of $x, y$ values with an outlier installed.

# %%
xs = jnp.linspace(-0.7, 0.7, 10)
ys = (
    -0.2
    + 0.4 * xs
    + 0.2 * xs**2
    + 0.05 * jax.random.normal(key=jax.random.PRNGKey(1), shape=xs.shape)
)
ys = ys.at[2].set(0.4)

Plot.dot({"x": xs, "y": ys})

# %% [markdown]
# We use a object called `CurveFit` to get an importance sample. Internally, `CurveFit` is written in terms of the map and switch combinators. That constructor will need simple GenerativeFunctions for the inlier and outlier models. I considered sticking with my design decision before of requiring the user to supply these in terms of the primitive distributions like `genjax.normal`, but instead, since the GFs are so simple, we may was well just write them down:

# %%


# %% [markdown]
# The alternative might have been to let the user say, for example,
# ```Python
# lambda y: genjax.normal(y, 0.1)
# ```
# in the first instance, and that would avoid presenting the decorator, `@`, and a couple of other things this notebook hasn't talked about yet.

# %%
p_outlier = genjax.beta(1.0, 1.0)
sigma_inlier = genjax.uniform(0.0, 0.3)
curve_fit = b.CurveFit(curve=quadratic, sigma_inlier=sigma_inlier, p_outlier=p_outlier)

# %% [markdown]
# We'll need a function to render the sample from the posterior: since, behind the scenes, Jax has turned the BlockFunctions into BlockFunctions of vectors of parameters, that code will be in terms of `tree_map`.


# %%
def plot_posterior(tr: genjax.Trace, xs: FloatArray, ys: FloatArray):
    return (
        plot_functions(tr.get_subtrace(("curve",)).get_retval(), opacity=0.2)  # type: ignore
        + Plot.dot({"x": xs, "y": ys, "r": 4})
    )


# %% [markdown]
# All that remains is to generate the sample. We select $K$ samples from a posterior categorical distribution taken over $N$ samples. On my home Mac, whose GPU is not accessible to GenJAX, I can get $N=100\mathrm{k}$ importance samples in a few seconds! Recall that on GenJAX interpreted, each of these took a substantial fraction of second. From there, we can plot, say, $K=100$ of these with alpha blending to visualize the posterior.

# %%
tr = curve_fit.importance_sample(xs, ys, 100000, 200)
plot_posterior(tr, xs, ys)

# %% [markdown]
# This is an excellent result, thanks to GenJAX, and I think indicative of what can be done with a DSL to temporarily shift the focus away from the nature of JAX. In this version of the model, the inlier sigma and probability were inference parameters of the model. Let's examine the distributions found by this inference:


# %% [markdown]
#
# Maybe we can stretch to accommodate a periodic sample:


# %%
def periodic_ex(F: b.Block, key=jax.random.PRNGKey(3)):
    ys = (
        (
            0.2 * jnp.sin(4 * xs + 0.3)
            + 0.02 * jax.random.normal(key=key, shape=xs.shape)
        )
        .at[7]
        .set(-0.7)
    )
    fs = b.CurveFit(
        curve=F, sigma_inlier=sigma_inlier, p_outlier=p_outlier
    ).importance_sample(xs, ys, 100000, 100)
    return plot_posterior(fs, xs, ys)


periodic_ex(quadratic)

# %% [markdown]
# Though the sample points are periodic, we supplied the degree 2 polynomial prior, and got reasonable results quickly. Before trying the Periodic prior, we might try degree 3 polynomials, and see what we get:

# %%
cubic = b.Polynomial(max_degree=3, coefficient_d=genjax.normal(0.0, 1.0))
periodic_ex(cubic)

# %% [markdown]
# **LGTM!**
# Now for periodic prior.
#
# (NB: your results may vary, but if you see some darker lines this is because the importance sampling is done *with replacement*)

# %%
periodic_ex(periodic, key=jax.random.PRNGKey(222))

# %% [markdown]
# Interesting! The posterior is full of good solutions but also contains a sprinkling of Nyquist-impostors!
#
# ...well, it used to. But changing the way the inlier_sigma is now inferred rather than prescribed has changed the quality of the posterior a great deal: Perhaps by allowing small sigmas, we have allowed a trace through which dominates all the others, weight-wise. We can get the old "curve-cloud" behavior back by fixing on 0.3 as the inlier sigma:


# %%
def periodic_ex2(F: b.Block):
    ys = (0.2 * jnp.sin(4 * xs + 0.3)).at[7].set(-0.7)
    fs = b.CurveFit(
        curve=periodic,
        sigma_inlier=genjax.uniform(0.1, 0.2),
        p_outlier=genjax.uniform(0.05, 0.2),
    ).importance_sample(xs, ys, 100000, 100)
    return plot_posterior(fs, xs, ys)


periodic_ex2(quadratic)

# %% [markdown]
# ## Conclusion

# %% [markdown]
# I like this framework more than I thought I would when I started it. A wise man once said it is easier to write probabilistic programming languages than probabilistic programs and I found it so. The experience of doing this helped me to understand JAX better. In particular, the idea of creating a custom `Pytree` object seemed exotic to me before I started this, but: if you want to have a Generative Function that produces something other than an array, while retaining full JAX-compatibility, it's exactly what you should do. In this case, the DSL allows the construction and composition of Generative Functions that sample from distributions of real-valued functions on the real line, and that's what curve fitting is about.
#
# ## Grand Finale: the ascending periodic example


# %% [markdown]
# Keeping track of the curve_fit, xs & ys, and other data for different
# experiments we will conduct can be confusing, so we'll write a little
# function that yokes the experiment material into a dict.
def ascending_periodic_ex(F: b.Block):
    xs = jnp.linspace(-0.9, 0.9, 20)
    ys = (0.7 * xs + 0.3 * jnp.sin(9 * xs + 0.3)).at[7].set(0.75)
    ys += jax.random.normal(key=jax.random.PRNGKey(22), shape=ys.shape) * 0.07
    curve_fit = b.CurveFit(curve=F, sigma_inlier=sigma_inlier, p_outlier=p_outlier)
    fs = curve_fit.importance_sample(xs, ys, 1000000, 100)
    return {"xs": xs, "ys": ys, "curve_fit": curve_fit, "tr": fs}


ascending_periodic_prior = quadratic + periodic
periodic_data = ascending_periodic_ex(ascending_periodic_prior)
plot_posterior(periodic_data["tr"], periodic_data["xs"], periodic_data["ys"])

# %% [markdown]
# The posterior distribution here is very thin, suggesting that the priors are too broad (note that I had to increase to 1M samples to get this far, which took 12.6s on my machine). Nonetheless, importance sampling on the sum function was able to find very plausible candidates.
#
# NB: actually that used to be true; now the posterior has a lot of interesting things in it (provoked in this instance I think by adding some noise to the y points)


# %%
def gaussian_drift(
    key: PRNGKey,
    curve_fit: b.CurveFit,
    tr: genjax.Trace,
    scale: ArrayLike = 2.0 / 100.0,
    n: int = 1,
):
    """Run `n` steps of the update algorithm. `scale` specifies the amount of gaussian drift in SD units."""
    if curve_fit.gf != tr.get_gen_fn():
        raise ValueError("The trace was not generated by the given curve")
    outlier_path = ("ys", ..., "outlier")

    def logit_to_prob(logit):
        return 1.0 / (1.0 + jnp.exp(-logit))

    def gaussian_drift_step(key: PRNGKey, tr: genjax.Trace):
        choices = tr.get_choices()

        def update(key: PRNGKey, cm: genjax.ChoiceMap) -> genjax.Trace:
            k1, k2 = jax.random.split(key)
            updated_tr, weight, arg_diff, bwd_problem = tr.update(k1, cm)
            mh_choice = jax.random.uniform(key=k2)
            ps = logit_to_prob(weight)
            return jax.lax.cond(mh_choice <= ps, lambda: updated_tr, lambda: tr)

        def update_coefficients(key: PRNGKey, coefficient_path: tuple):
            k1, k2 = jax.random.split(key)
            values = choices[coefficient_path]  # pyright: ignore [reportIndexIssue]
            drift = values + scale * jax.random.normal(k1, values.shape)

            # substitute ... in coefficient path with array of integer indices
            def fix(e):
                return jnp.arange(len(drift)) if e is Ellipsis else e

            new_path = tuple(fix(e) for e in coefficient_path)
            cm = C[new_path].set(drift)
            return update(k2, cm)

        def update_sigma_inlier(key: PRNGKey):
            k1, k2 = jax.random.split(key)
            s = choices["sigma_inlier"]  # pyright: ignore [reportIndexIssue]
            return update(
                key, C["sigma_inlier"].set(s + scale * jax.random.normal(key=k1))
            )

        def update_p_outlier(key: PRNGKey):
            # p_outlier is "global" to the model. Each point in the curve data
            # has an outlier status assignment. We can therefore measure the
            # posterior p_outlier and use that data to propose an update.
            k1, k2 = jax.random.split(key)
            outlier_states = choices[outlier_path]  # pyright: ignore [reportIndexIssue]
            print('o1', outlier_states.shape)
            n_outliers = jnp.sum(outlier_states)
            new_p_outlier = jax.random.beta(
                k1,
                1.0 + n_outliers,
                1 + len(outlier_states) + n_outliers,
            )
            return update(k2, C["p_outlier"].set(new_p_outlier))

        def update_outlier_state(key: PRNGKey):
            # this doesn't work: see GEN-324
            k1, k2 = jax.random.split(key)
            outlier_states = choices[outlier_path]  # pyright: ignore [reportIndexIssue]]
            print('o2', outlier_states.shape)
            flips = jax.random.bernoulli(k1, shape=outlier_states.shape)
            return update(
                k2,
                C["ys", jnp.arange(len(flips)), 'outlier'].set(
                    jnp.logical_xor(outlier_states, flips).astype(int)
                )
            )

        k1, k2, *ks = jax.random.split(key, 2 + len(curve_fit.coefficient_paths))
        for k, path in zip(ks, curve_fit.coefficient_paths):
            tr = update_coefficients(k, path)
        tr = update_p_outlier(k1)
        tr = update_sigma_inlier(k2)
        # tr = update_outlier_state(k3)  # GEN-324
        return tr

    # The first step is an empty update to stabilize the type of the trace (GEN-306)
    tr0 = jax.vmap(lambda t: t.update(jax.random.PRNGKey(0), C.d({})))(tr)[0]
    # The blocks library arranges for importance samples to come from vmap (even if
    # only one was requested), so we may expect a normally-scalar field like score
    # to have a leading axis which equals the number of traces within.
    K = tr.get_score().shape[0]  # pyright: ignore [reportAttributeAccessIssue]
    # This is a two-dimensional JAX operation: we scan through `n` steps of `K` traces
    sub_keys = jax.random.split(key, (n, K))
    tr, _ = jax.lax.scan(
        lambda trace, keys: (jax.vmap(gaussian_drift_step)(keys, trace), None),
        tr0,
        sub_keys,
    )
    return tr


# %% [markdown]
# In this cell, we will take the original curve fit size-200 importance sample of
# the degree-2 polynomial prior and run it through 100 steps of gaussian drift.
key, sub_key = jax.random.split(jax.random.PRNGKey(314159))
tru = gaussian_drift(sub_key, curve_fit, tr, n=100)
plot_posterior(tru, xs, ys)

# %% [markdown]
# Now let's try something more difficult. We will gaussian drift the
# periodic example.
key, sub_key = jax.random.split(key)
periodic_t1 = gaussian_drift(
    sub_key, periodic_data["curve_fit"], periodic_data["tr"], n=100
)
plot_posterior(periodic_t1, periodic_data["xs"], periodic_data["ys"])
# %% [markdown]
# Let's drift the _result_ of that experiment at a smaller scale and see if that helps
key, sub_key = jax.random.split(key)
periodic_t2 = gaussian_drift(
    sub_key, periodic_data["curve_fit"], periodic_t1, n=100, scale=0.005
)
plot_posterior(periodic_t2, periodic_data["xs"], periodic_data["ys"])

# %%
