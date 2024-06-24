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
# pyright: reportUnusedExpression=false
# %% [markdown]
# # DSL for curve fit inference
#
# As part of the `genjax.interpreted` performance investigation, I wanted to investigate a DSL for the curve-fitting task which could achieve JAX-accelerated performance without introducing JAX concepts such as the Switch and Map combinators, `in_axes`, and other things that might complicate the exposition of inference for the newcomer. While doing so, I also studied ways to "automatically" thread randomness through the computation without having to discuss the careful use of `jax.random.split` which we recommend to GenJAX users. Having done the experiment, I have mixed feelings about the results: on the one hand, it is possible to get JAX-level performance with curve-building combinators, but the price of so doing is that the GFI is hidden from view as well, and that may be too far a step. Nonetheless, if you're still interested in what can be achieved in this framework, read on!

# %%
import genjax
from genjax import ChoiceMapBuilder as C
from genjax.typing import PRNGKey, FloatArray, ArrayLike
import genstudio.plot as Plot
import jax
import jax.numpy as jnp
import jax.tree
from blocks import (
    Block,
    BlockFunction,
    CoinToss,
    CurveFit,
    Exponential,
    Periodic,
    Polynomial,
)

genjax.pretty()

# %% [markdown]
# The `blocks` library is concentrated on a simple $\mathbb{R}\rightarrow\mathbb{R}$ inference problem, localized to the unit square for convenience, as found in the introductory GenJAX notebooks. We provide a "basis" of polynomial, $ae^{bx}$, and $a\sin(\phi + 2\pi x/T)$ functions, each of whose parameters are drawn from a standard distribution that the user supplies. The intro curve fit task contemplates a polynomial of degree 2 with normally distributed coefficients, which we may write as

# %%
P = Polynomial(max_degree=2, coefficient_d=genjax.normal(0.0, 1.0))

# %% [markdown]
# Note one felicity we have achieved already: we can specify the normal distribution in a simple fashion without having to tuple the arguments or provide randomness: that comes later. The type of `P` is `Block`, a name inspired by [blockly](https://developers.google.com/blockly). The only operation we can do on a Block is request an array of samples from it.

# %%
p = P.sample()
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
def plot_functions(fns: BlockFunction, **kwargs):
    xs = jnp.linspace(-1, 1, 200)
    yss = jax.vmap(fns)(xs)
    return Plot.new(
        [
            Plot.line({"x": xs, "y": ys, "stroke": i % 12}, kwargs)
            for i, ys in enumerate(yss.T)
        ],
        {"clip": True, "height": 400, "width": 400, "y": {"domain": [-1, 1]}},
    )


def plot_priors(B: Block, n: int):
    return plot_functions(B.sample(n).get_retval())


plot_priors(P, 100)

# %% [markdown]
# We can do the same for our Periodic and Exponential distributions:

# %%
Q = Periodic(
    amplitude=genjax.beta(2.0, 5.0),
    phase=genjax.uniform(-1.0, 1.0),
    period=genjax.normal(1.0, 1.0),
)

R = Exponential(a=genjax.normal(0.0, 1.0), b=genjax.normal(0.0, 1.0))

# %%
plot_priors(Q, 50)

# %%
plot_priors(R, 50)

# %% [markdown]
# `Block` objects support the pointwise operations $+, *$ as well as `@` (function composition). Vikash's favorite ascending-periodic function might be modeled by the sum of a polynomial and a periodic function which we can write simply as $P+Q$:

# %%
plot_priors(P + Q, 15)

# %% [markdown]
# It does seem like the goal function lies in the span of the prior distribution in this case. (I pause here to note that pointwise binary operations in `Block` are not commutative as you might expect, because the randomness supplied by `sample` is injected left to right).
#
# The binary operations produce traces representing the expression tree that created them:

# %%
(P + Q).sample().get_choices()

# %% [markdown]
# Even if we don't appear to be using the GFI, we still have traces. The final step in this demo is to conduct an importance sample at JAX speed given a set of points. It's at this point that we want to create a model that supports the concept of outlier. In the original examples, the model tossed a Bernoulli coin, and selected between polynomials with a loose and tight variance. Vikash has elsewhere proposed switching between a normal and uniform distribution in this case, which we will attempt here. This is the point at which we run into the "if problem" of JAX, which we finesse using a `CoinToss` block (a better name might be `BernoulliChoice`, but we're just sketching here.)

# %%
ct = CoinToss(
    probability=0.2,
    heads=Polynomial(max_degree=1, coefficient_d=genjax.normal(0.0, 0.05)),
    tails=Polynomial(max_degree=1, coefficient_d=genjax.normal(1.0, 0.05)),
)

# %%
plot_priors(ct, 50)

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
curve_fit = CurveFit(curve=P, sigma_inlier=sigma_inlier, p_outlier=p_outlier)

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


# %%
def param_histo(tr):
    return Plot.autoGrid(
        [
            Plot.histogram(tr.get_choices()["sigma_inlier"]),
            Plot.histogram(tr.get_choices()["p_outlier"]),
        ]
    )


param_histo(tr)


# %% [markdown]
# While we're at it, we can generate summary statistics on the various parameters, to improve the fit.

# %%
tr.get_choices()["curve", "p", ..., "coefficients"]

# %%
coefficients = tr.get_choices()["curve", "p", ..., "coefficients"]
means = jnp.mean(coefficients, axis=0)
stdevs = jnp.sqrt(jnp.var(coefficients, axis=0))
means, stdevs

# %%
tr

# %% [markdown]
# equipped with that information, we could make a better prior:

# %%
updated_distributions = list(map(genjax.normal, means, stdevs))
# P2 = Polynomial(max_degree=2, coefficient_d=[genjax.normal(means[0], stdevs[0]), genjax.normal(means[0], stdevs[0]), genjax.normal(means[1], stdevs[1])])
updated_distributions

# %% [markdown]
#
# Maybe we can stretch to accommodate a periodic sample:


# %%
def periodic_ex(F: Block, key=jax.random.PRNGKey(3)):
    ys = (
        (
            0.2 * jnp.sin(4 * xs + 0.3)
            + 0.02 * jax.random.normal(key=key, shape=xs.shape)
        )
        .at[7]
        .set(-0.7)
    )
    fs = CurveFit(
        curve=F, sigma_inlier=sigma_inlier, p_outlier=p_outlier
    ).importance_sample(xs, ys, 100000, 100)
    return plot_posterior(fs, xs, ys)


periodic_ex(P)

# %% [markdown]
# Though the sample points are periodic, we supplied the degree 2 polynomial prior, and got reasonable results quickly. Before trying the Periodic prior, we might try degree 3 polynomials, and see what we get:

# %%
periodic_ex(Polynomial(max_degree=3, coefficient_d=genjax.normal(0.0, 1.0)))

# %% [markdown]
# **LGTM!**
# Now for periodic prior.
#
# (NB: your results may vary, but if you see some darker lines this is because the importance sampling is done *with replacement*)

# %%
periodic_ex(Q, key=jax.random.PRNGKey(222))

# %% [markdown]
# Interesting! The posterior is full of good solutions but also contains a sprinkling of Nyquist-impostors!
#
# ...well, it used to. But changing the way the inlier_sigma is now inferred rather than prescribed has changed the quality of the posterior a great deal: Perhaps by allowing small sigmas, we have allowed a trace through which dominates all the others, weight-wise. We can get the old "curve-cloud" behavior back by fixing on 0.3 as the inlier sigma:


# %%
def periodic_ex2(F: Block):
    ys = (0.2 * jnp.sin(4 * xs + 0.3)).at[7].set(-0.7)
    fs = CurveFit(
        curve=Q,
        sigma_inlier=genjax.uniform(0.1, 0.2),
        p_outlier=genjax.uniform(0.05, 0.2),
    ).importance_sample(xs, ys, 100000, 100)
    return plot_posterior(fs, xs, ys)


periodic_ex2(P)

# %% [markdown]
# ## Conclusion

# %% [markdown]
# I like this framework more than I thought I would when I started it. A wise man once said it is easier to write probabilistic programming languages than probabilistic programs and I found it so. The experience of doing this helped me to understand JAX better. In particular, the idea of creating a custom `Pytree` object seemed exotic to me before I started this, but: if you want to have a Generative Function that produces something other than an array, while retaining full JAX-compatibility, it's exactly what you should do. In this case, the DSL allows the construction and composition of Generative Functions that sample from distributions of real-valued functions on the real line, and that's what curve fitting is about.
#
# ## Grand Finale: the ascending periodic example


# %%
def ascending_periodic_ex(F: Block):
    xs = jnp.linspace(-0.9, 0.9, 20)
    ys = (0.7 * xs + 0.3 * jnp.sin(9 * xs + 0.3)).at[7].set(0.75)
    ys += jax.random.normal(key=jax.random.PRNGKey(22), shape=ys.shape) * 0.07
    fs = CurveFit(
        curve=F, sigma_inlier=sigma_inlier, p_outlier=p_outlier
    ).importance_sample(xs, ys, 1000000, 100)
    return plot_posterior(fs, xs, ys)


ascending_periodic_ex(P + Q + R)

# %% [markdown]
# The posterior distribution here is very thin, suggesting that the priors are too broad (note that I had to increase to 1M samples to get this far, which took 12.6s on my machine). Nonetheless, importance sampling on the sum function was able to find very plausible candidates.
#
# NB: actually that used to be true; now the posterior has a lot of interesting things in it (provoked in this instance I think by adding some noise to the y points)


# %%
def gaussian_drift(
    key: PRNGKey, curve_fit: CurveFit, tr: genjax.Trace, scale: ArrayLike = 2.0 / 100.0, n: int = 1
):
    if curve_fit.gf != tr.get_gen_fn():
        raise ValueError("The trace was not generated by the given curve")
    coefficient_path = ("curve", "p", ..., "coefficients")
    # coefficient_paths = curve_fit.address_segments()
    p_outlier_path = ("p_outlier",)
    outlier_path = ("ys", ..., "outlier")
    display_width = 2.0
    sd = display_width / 100.0

    def logit_to_prob(logit):
        return 1.0 / (1.0 + jnp.exp(-logit))

    # just had a thought: am I doing this the wrong way? Would it make sense to
    # write this in a way where we update ONE particle, and vmap that? Currently,
    # we're producing our choicemaps pre-vmapped, but maybe that is a mistake.


    @jax.jit
    def gaussian_drift_step(key: PRNGKey, tr: genjax.Trace):
        choices = tr.get_choices()
        #shape = choices[coefficient_path].shape  # pyright: ignore [reportIndexIssue]

        def update(
            key: PRNGKey, original: genjax.ChoiceMap, cm: genjax.ChoiceMap
        ) -> genjax.Trace:
            k1, k2, k3 = jax.random.split(key, 3)
            (updated_tr, weights, arg_diff, bwd_problem) = jax.vmap(
                lambda k, t, c: t.update(k, c)
            )(jax.random.split(k1, shape[:1]), tr, cm)
            mh_choices = jax.random.uniform(key=k2, shape=shape[:1])
            ps = logit_to_prob(weights)
            winners = jnp.expand_dims(mh_choices < ps, axis=1)
            # winners.shape is (N, 1), which is broadcastable to the arrays in the vmapped trace
            filtered_update = jax.tree.map(
                lambda updated, original: jnp.where(winners, updated, original),
                cm,
                original,
            )
            (scored_tr, weights, arg_diff, bwd_problem) = jax.vmap(
                lambda k, t, c: t.update(k, c)
            )(jax.random.split(k3, shape[:1]), tr, filtered_update)
            return scored_tr

        def update_coefficients(key: PRNGKey):
            key, k1, k2 = jax.random.split(key, 3)
            values = choices[coefficient_path]  # pyright: ignore [reportIndexIssue]
            to_choicemap = jax.vmap(
                lambda v: C[
                    "curve", "p", jnp.arange(len(v), dtype=int), "coefficients"
                ].set(v)
            )
            drift = values + sd * jax.random.normal(k1, shape)
            return update(k2, to_choicemap(values), to_choicemap(drift))

        def update_p_outlier(key: PRNGKey):
            # p_outlier is "global" to the model. Each point in the curve data
            # has an outlier status assignment. We can therefore measure the
            # posterior p_outlier and use that data to propose an update.
            k1, k2 = jax.random.split(key)
            p_outliers = choices[p_outlier_path]  # pyright: ignore [reportIndexIssue]
            n_outliers = jnp.sum(choices[outlier_path], axis=1)  # pyright: ignore [reportIndexIssue]
            new_p_outliers = jax.random.beta(
                k1,
                1.0 + n_outliers,
                1 + len(n_outliers) + n_outliers,
                shape=n_outliers.shape,
            )
            to_choicemap = jax.vmap(lambda p: C["p_outlier"].set(p))
            return update(k2, to_choicemap(p_outliers), to_choicemap(new_p_outliers))

        def update_outlier_state(key: PRNGKey):
            # we aren't doing this one for two reasons:
            # a) there may be a genjax bug (we get a stack trace when we try) and
            # b) the instructions ("update the outlier indicator for each variable
            # using an MH proposal that proposes to flip each datapoint (from inliner
            # to outlier, or vice versa) with probability 0.5, or otherwise keeps
            # it the same") is the same thing as just changing all the outlier states
            # to a new random 0.5 bernoulli coin flip.
            def to_choicemap(v):
                return jax.vmap(
                    lambda o: C["ys", jnp.arange(len(o), dtype=int), "outlier"].set(o)
                )(v)

            k1, k2 = jax.random.split(key)
            outlier_states = choices[outlier_path]  # pyright: ignore [reportIndexIssue]
            flips = jax.random.bernoulli(k1, shape=outlier_states.shape)
            return update(
                k2,
                to_choicemap(outlier_states),
                to_choicemap(jnp.logical_xor(outlier_states, flips).astype(int)),
            )

        k1, k2, k3 = jax.random.split(key, 3)
        tr = update_coefficients(k1)
        tr = update_p_outlier(k2)
        # tr = update_outlier_state(k3)
        return tr

    # gaussian_drift_step has the effect of slightly alterning the type
    # of objects within the trace, but once it has been applied, the
    # typing is stable. Therefore, we seed the accelerated iteration with
    # the result of the first update step, and use scan to iterate over
    # the remaining n-1 steps
    sub_keys = jax.random.split(key, n)
    tr, _ = jax.lax.scan(
        lambda trace, key: (gaussian_drift_step(key, trace), None),
        gaussian_drift_step(sub_keys[0], tr),
        sub_keys[1:],
    )
    return tr


# %%
key, sub_key = jax.random.split(jax.random.PRNGKey(314159))
tru = gaussian_drift(key, curve_fit, tr, n=100)
plot_posterior(tru, xs, ys)

# %%
key, sub_key = jax.random.split(jax.random.PRNGKey(314159))
tru.get_gen_fn() == curve_fit.gf
#plot_posterior(gaussian_drift(sub_key, tru, n=100, scale=.005), xs, ys)

# %%
tr
# %%
tr.get_choices()['ys',...,'outlier']
# %%
def ap_ex2(F: Block, key=jax.random.PRNGKey(0)):
    xs = jnp.linspace(-0.9, 0.9, 20)
    ys = (0.7 * xs + 0.3 * jnp.sin(9 * xs + 0.3)).at[7].set(0.75)
    key, sub_key = jax.random.split(key)
    ys += jax.random.normal(key=sub_key, shape=ys.shape) * 0.07
    fs = CurveFit(
        curve=F, sigma_inlier=sigma_inlier, p_outlier=p_outlier
    ).importance_sample(xs, ys, 100000, 100)
    return fs
    key, sub_key = jax.random.split(key)
    fs = gaussian_drift(sub_key, fs, n=100)
    return plot_posterior(fs, xs, ys)

fs = ap_ex2(P+Q)

# %%
chs = fs.get_choices()

# %
# %%
chs['curve','l','p',...,'coefficients']
chs['curve','r','T']
# %%
# %%
list((P+Q).address_segments())
list(P.address_segments())# %%
# %%
