# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
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
    # %pip install --quiet keyring keyrings.google-artifactregistry-auth  # type: ignore # noqa
    # %pip install --quiet genjax-blocks==0.1.0.post8.dev0+a289d26 --extra-index-url https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/  # type: ignore # noqa

# %% [markdown]
# # Curve fitting via distributions and inference
#
# This tutorial presents an introduction to solving problems using distributions and inference, working with the example of fitting curves to data.
#
# [Comment on Colin's abstractions on top of GenJAX?]

# %%
import genjax
import functools
from genjax import ChoiceMapBuilder as C
from genjax.typing import PRNGKey, FloatArray, ArrayLike
import genjax_blocks as b
import genstudio.plot as Plot
import jax
import jax.numpy as jnp
import penzai.pz as pz

# %% [markdown]
# ## Distributions
#
# Probability distributions are a mathematical construct used to express uncertain quantities.
#
# We will represent them as code objects that generate possible values for these quantities according to how likely they are.

# %% [markdown]
# ### Distributions over numbers
#
# When the uncertain quantities are numbers, there are many standard examples (will use here: normal, uniform, beta).  Note that these are code objects that can be queried for samples.

# %%
u = genjax.uniform(0.0, 1.0)

# Colin: how to brush randomness under the rug?  wishing for u.sample()
key = jax.random.PRNGKey(8327)
key, subkey = jax.random.split(key)
u.simulate(key=subkey, args=()).get_retval()

# %% [markdown]
# The relative frequencies of their samples can be pictured through their *density functions*.

# %%
# make a plot

Plot.histogram(u.repeat(n=1000).simulate(key=subkey, args=()).get_retval())
# ... can't we just monadically thread the random key somehow?

# %% [markdown]
# ### Distributions over curves
#
# The uncertain quantity expressed by a distribution can be any kind of mathematical gadget.  Our primary example will be distributions over curves, that is, graphs functions $f(x)$.  Here are some basic examples.
#
# First example: curves of the form $f(x) = a e^{b x}$, where $a$ and $b$ *each vary through their own given distributions*.

# %%
exponential = b.Exponential(a=genjax.normal(0.0, 1.0), b=genjax.normal(0.0, 1.0))

exponential2 = b.Exponential(a=genjax.normal(1.0,0.1), b = genjax.normal(1.0, 0.1))

# plot some samples...

( b.plot_functions(exponential.sample(100).get_retval())
 & b.plot_functions(exponential2.sample(100).get_retval()) )

# %% [markdown]
# Similarly for sinusoidal functions $f(x) = a \sin(2\pi\omega\,(x + \varphi))$.

# %%
periodic = b.Periodic(
    amplitude=genjax.beta(2.0, 5.0),
    phase=genjax.uniform(0.0, 1.0),
    frequency=genjax.normal(1.0, 1.0),
)

# plot...

b.plot_functions(periodic.sample(25).get_retval())

# %% [markdown]
# Similarly for polynomial for a *fixed max degree*, and the same distribution across all the coeffs.

# %%
# Colin: do we care that quadratic has a different coeff std.dev.?
constant = b.Polynomial(max_degree=0, coefficient_d=genjax.normal(0.0, 2.0))
linear = b.Polynomial(max_degree=1, coefficient_d=genjax.normal(0.0, 2.0))
quadratic = b.Polynomial(max_degree=2, coefficient_d=genjax.normal(0.0, 2.0))
cubic = b.Polynomial(max_degree=3, coefficient_d=genjax.normal(0.0, 2.0))
quartic = b.Polynomial(max_degree=4, coefficient_d=genjax.normal(0.0, 2.0))

functools.reduce(lambda a, b: a & b, [
    b.plot_functions(f.sample(25).get_retval())
    for f in [constant, linear, quadratic, cubic, quartic]
])
# %% [markdown]
# ### Combining distributions over curves
#
# We can combine distributions over curves to take new ones.
#
# Operators `+` ("sum"), `*` ("product"), `@` ("composite") take two distributions over curves and produce a new one.  The new distribution first draws sample curves from the two operand distributions, then returns as its sample the curve whose function is the pointwise sum, pointwise product, or composite function of the two drawn curves.

# %%
wiggly_line = linear + periodic
swoosh = periodic @ quadratic
zap = exponential * (periodic @ exponential)

functools.reduce(lambda a, b: a & b, [
    b.plot_functions(f.sample(25).get_retval())
    for f in [wiggly_line, swoosh, zap]
])

# %% [markdown]
# ## Fitting curves to data: optimization
#
# The problem is widely familiar: from algebra on up, students fit curves to data by backsolving for the coefficients in a function.
#
# We are going to work in a context where the fit is not exact, only approximate.  Maybe there does not exist an exact fit, such as an overdetermined system.  Or there is one, and finding it exactly is computationally out of reach.  Or we can find one, possibly many fits, but they are all rather unlikely or wackly, like an extremely high-degree polynomial to approximate periodic data.
#
# In another light, only expecting an approximate fit is asking for something weaker, and is a *simplifying* relaxation of the problem.

# %% [markdown]
# ### Noisy curves
#
# The `NoisyCurve` object represents a distribution over datasets.  It takes in a curve $f(x)$ and a noise level $\sigma$.  When queried for an sample on the inputs $x_i$ for $i=1,\ldots,N$, it independently for each $i=1,\ldots,N$ draws a sample $y_i$ for the normal distribution centered at $f(x_i)$ with standard deviation $\sigma$.
#
# We will consider these as a model for data sets that do not exactly lie on curves.

# %%
# Example of some NoisyCurve samples = finite point sets, drawn from a single curve at a time

xs = jnp.linspace(-0.7, 0.7, 10)

sample_curve = quartic.sample(k=jax.random.PRNGKey(134)).get_retval()
ys_latent = sample_curve(xs)

noisy_data_model = b.NoisyData(sigma_inlier=genjax.uniform(0.0, 0.1))
noisy_ys_samples = noisy_data_model.sample(ys_latent, 12)
print(f"sigma_in values: {noisy_ys_samples.get_choices()["kernel_params", "σ_inlier"]}")
noisy_ys_values = noisy_ys_samples.get_retval()

Plot.new(
    [Plot.line(list(zip(xs, ys_latent)))] +
    [Plot.dot(list(zip(xs, ys)), stroke=i) for i, ys in enumerate(noisy_ys_values)])

# %% [markdown]
# ### Classic technique: least squares
#
# Suppose we are trying to fit data $(\vec x_i,\vec y_i)$ for $i=1,\ldots,N$ where the $\vec x_i,\vec y_i$ are vectors, using a linear equation $\vec y = M \vec x + \vec b$ where $M$ is a matrix and $\vec b$ is a vector.  When $N$ is large and the system is overdetermined, how do we make sense of the situation?
#
# A common answer is to choose the $M$ and $\vec b$ that minimize the sum of the squared errors $\sum_{i=1}^N \|M\vec x_i + \vec b - \vec y_i\|^2$.  With a little elbow grease, this can be explicitly solved, and libraries do it for us.

# %%
ys = noisy_ys_values[0]

# First, reform the xs = [..., x_i, ...] into a matrix whose rows are the augmented vectors [1, x_i].
xs_augmented = jnp.vstack([jnp.ones(len(xs)), xs]).T

# Find the least squares fit to the system of equations
# [c0, c1] dot [1, x_i] == c0 + c1 x_i == y_i.
c0, c1 = jnp.linalg.lstsq(xs_augmented, ys)[0]
line = lambda x: c0 + c1*x

graph_xs = jnp.array([xs[0], xs[-1]])
Plot.new([
    Plot.dot(list(zip(xs, ys))),
    Plot.line(list(zip(graph_xs, line(graph_xs))))
])
# %% [markdown]
# Sometimes least squares fitting may be hijacked to solve other problems.  For instance, suppose we wanted to fit a polynomial curve of fixed degree $d$ to some data $(x_i,y_i)$ for $i=1,\ldots,N$.  The right hand side of the desired equation $y = a_d x^d + a_{d-1} x^{d-1} + \cdots + a_1 x + a_0$ may be a polynomial in $x$, but it is a *linear function of the powers of $x$*.  Therefore we can perform least squares fitting on the data $(\vec x_i,y_i)$ where $\vec x_i$ is the vector of powers of $x_i$.

# %%
# Form the matrix whose rows are the power-vectors [1, x_i, x_i**2, ..., x_i**d]...
def powers_vector(x, max_degree):
    return jnp.pow(jnp.array(x)[jnp.newaxis].T, jnp.arange(max_degree + 1))
# ...with max_degree=4 (quartic)
xs_powers = powers_vector(xs, 4)

# Find the least squares fit to the system of equations
# [c_0, c_1, ..., c_d] dot [1, x_i, x_i**2, ..., x_i**d] == c_0 + c_1 * x_i + c_2 * x_i**2 + ... + c_d * x_i**d == y_i.
cs = jnp.linalg.lstsq(xs_powers, ys)[0]
poly_f = lambda x: powers_vector(x, 4) @ cs

graph_xs = jnp.linspace(xs[0], xs[-1], 3 * xs.shape[0])
Plot.new([
    Plot.dot(list(zip(xs, ys))),
    Plot.line(list(zip(graph_xs, poly_f(graph_xs))))
])
# %% [markdown]
# ### Gradient descent via noisy curves
#
# The exact solution of least squares breaks down in the non-linear setting, however: the reader is invited to struggle with adapting it to fitting sinusoidal curves to data!  It is then common to approximately optimize the sum-squared error using *gradient descent*.
#
# Conveniently, the log density of a sample $(y_i)_i$ under one or our noisy curve distributions is the sum of the corresponding log normal densities of the $y_i$, which are in turn proportional to the squared errors $(f(x_i)-y_i)^2$.  Thus optimizing the log density of a dataset under this distribution *for varying curves* is equivalent to optimizing the sum-squared error.
#
# Let's try to fit the following points with a member of the exponential family.

# %%
nonlinear_sample_curve = exponential.sample(k=jax.random.PRNGKey(1)).get_retval()
params_latent = nonlinear_sample_curve.params[0]
print(f"Latent parameters: [a, b] = {params_latent}")
ys_latent = nonlinear_sample_curve(xs)

ys_observed = noisy_data_model.sample(ys_latent).get_retval()[0]

Plot.new([
    Plot.line(list(zip(xs, ys_latent)), strokeDasharray="7"),
    Plot.dot(list(zip(xs, ys_observed))),
])

# %%
joint_model = b.CurveDataModel(exponential, noisy_data_model)

a_guess, b_guess = -1.0, -1.0
params_guess = jnp.array([a_guess, b_guess])

sigma_in = 0.05
jitted_grad = jax.jit(jax.jacfwd(lambda params: joint_model.log_density(params, sigma_in, xs, ys_observed)))

learning_rate = 1e-5
params_optimized = params_guess
N_steps = 1000
for _ in range(N_steps):
    grad = jitted_grad(params_optimized)
    params_optimized = params_optimized + learning_rate * grad

curve_optimized = exponential.curve_from_params(params_optimized)

Plot.new([
    Plot.line(list(zip(xs, curve_optimized(xs)))),
    Plot.dot(list(zip(xs, ys_observed))),
])

# %% [markdown]
# ### The problem of outliers
#
# Blind optimization suffers from sensitivity to outliers in the data.

# %%
# Example of an outlier throwing off an optimization

# %% [markdown]
# We can instead imagine curves that produce noisy data *including some outliers*.  This intuition can be simply codified into a more sophisticated distribution.

# %%
# NoisyOutliersCurve object
# Example of some NoisyOutliersCurve samples = finite point sets, drawn from a single curve at a time

# %% [markdown]
# We again get a reasonable notion of how good a fit is (density in the data model).
#
# How to optimize in non-differentiable context?

# %% [markdown]
# ### The issue of multiple modes
#
# A subtler issue lurks our methodology: the assumption has already been baked in that the answer to the fitting problem consists of a single best curve.
#
# For example, the same data can be perfectly fit by curves of two very different parameters:

# %%
# Example of a data set plus a couple very different looking BlockFunctions (maybe same Block but different params?) that fit the data well.

# %% [markdown]
# What kind of solutions to the fitting problem should we even be looking for, that reflect the diversity of acceptable answers?

# %% [markdown]
# ## Fitting curves to data: conditioning
#
# Probability theory is a language that allows a precise form of reasoning about uncertain values.  In short, the object that soundly answers the curve fitting problem is a *probability distribution over curves*, expressing which curves were more likely to have generated the data.  Then our particular answers of best-fit curves are simply the (possibly multitudinous) relatively high-probability samples from this distribution.
#
# ### Bayesian reasoning
#
# Bayesian reasoning is a framework that starts with the following assumptions.
# * The first input is a *prior distribution* over hypotheses, which expresses which ones we are willing to consider, and how relatively willing we are to consider them, in advance of seeing the data set.  We package all of our knowledge or ignorance about which hypotheses will be relevant into this.
#   *  In our case study, the hypotheses are curves, and each Block expresses a prior over curves.
# * The second input is called the *likelihood kernel*, and it expresses how likey any observed data should be, given which hypothesis holds.  In other words, it is a function from hypotheses to probability distributions over data.
#   *  In our case study, a likelihood is supplied by one of our data models, with just noise, or noise plus outliers.
# * The third input is the new information of the *data*.
#
# Given the fact that we have observed the data, how ought we update our beliefs about our hypotheses?  What new probability distribution, call it the *posterior distribution*, over hypotheses expresses these updated beliefs?  In order to answer, we make the following constructions.
# * The prior and the likelihood assemble into a *joint distribution* which expresses the total probability of an experiment in which both a hypothesis holds and the particular data were observed.
#   * Numerically it simply computes the product of the prior probability and the likelihood probability.
# * The joint distribution is used to define the *marginal distribution*, which is a probability distribution over the data.  It expresses our total belief of how likely we should believe a data will be observed, upon averaging (according to our prior beliefs) over all hypotheses that might lead to them being observed.
#   * Numerically it is a sum/integral over hypotheses with respect to the prior.
# * The joint and the marginal are together used to define the *conditional distribution* over hypotheses given the data.  The conditional density of some hypothesis given some observed data is proportional to the joint probability of the two.  It simply needs to be renormalized so that the total probability of the hypotheses with the given data is again $1$.  This works out to be:
#   * Numerically, the conditional probability of a hypothesis is equal to its joint probability together with the data, divided by the marginal probability of the data.
#
# The Bayesian view is that the posterior distribution is none other than the conditional distribution.
#
# The art of *implementing* conditioning is what we call *Bayesian inference*.  In particular, generating samples from the conditional distribution is precisely the same as inferring which hypotheses are good fits to the observations.
#
# In general, exact inference is computationally hard-to-impossible, and better approxiamations require more compute; herein lies the ProbProgrammer's design space.

# %% [markdown]
# ### Inference 1: needle in haystack
#
# First generate plausible but probabilistically skewed data, then must correct them by resampling.
#
# One might ask why not just take the single best fit.  Many reasons, starting with multi-modality, and eventually "seeing" the whole space of answers.

# %%
# Importance sampling examples

# %% [markdown]
# ### Inference 2: improving our guesses
#
# We can do much better by exploiting the known structure of the problem.  Starting from a decent guess, we can explore nearby it for better fits.

# %%
# Gaussian drift

# %%
# Gradient-biased Gaussian drift

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
import penzai.pz as pz

# %% [markdown]
# The `blocks` library is concentrated on a simple $\mathbb{R}\rightarrow\mathbb{R}$ inference problem, localized to the unit square for convenience, as found in the introductory GenJAX notebooks. We provide a "basis" of polynomial, $ae^{bx}$, and $a\sin(2\pi\omega\,(x + \varphi))$ functions, each of whose parameters are drawn from a standard distribution that the user supplies. The intro curve fit task contemplates a polynomial of degree 2 with normally distributed coefficients, which we may write as

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
def plot_functions(fns: b.BlockFunction, winningIndex=None, **kwargs):
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

def plot_priors(B: b.Block, n: int, **kwargs):
    return plot_functions(B.sample(n).get_retval(), **kwargs)

# yss = f_i(x_j) (??)
# want: 100 rows of 200 pairs of (x,y) ?

plot_priors(quadratic, 100)

# %% [markdown]
# We can do the same for our Periodic and Exponential distributions:

# %%
periodic = b.Periodic(
    amplitude=genjax.beta(2.0, 5.0),
    phase=genjax.uniform(0.0, 1.0),
    frequency=genjax.normal(1.0, 1.0),
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
# We'll need a function to render the sample from the posterior. We can extract the
# return value of the trace named curve to get the vector of functions; this has the
# same shape as samples from the prior, so `plot_functions` works here too.
# %%
def plot_posterior(tr: genjax.Trace, xs: FloatArray, ys: FloatArray, **kwargs):
    ch = tr.get_choices()
    outliers = ch['ys', ..., 'outlier']
    outlier_fraction = jnp.sum(outliers, axis=0) / outliers.shape[0]
    data = list(zip(xs, ys, outlier_fraction))
    plot = (
        plot_functions(tr.get_subtrace(("curve",)).get_retval(), opacity=0.2, **kwargs)  # type: ignore
        + Plot.new(Plot.dot(data, fill=Plot.js('(d) => d3.interpolateViridis(d[2])'), r=4))
    )
    return plot
# %% [markdown]
# All that remains is to generate the sample. We select $K$ samples each drawn from a posterior
# categorical distribution taken over $N$ samples. On my home Mac, whose GPU is not
# accessible to GenJAX, I can get $KN=100\mathrm{k}$ importance samples in a few seconds!
# Recall that on GenJAX interpreted, each of these took a substantial fraction of second.
# From there, we can plot, say, $K=100$ of these with alpha blending to visualize the posterior.

# %%
tr = curve_fit.importance_sample(xs, ys, 4000, 25)
plot_posterior(tr, xs, ys)

# %% [markdown]
# This is an excellent result, thanks to GenJAX, and I think indicative of what can be done with a DSL to temporarily shift the focus away from the nature of JAX. In this version of the model, the inlier sigma and probability were inference parameters of the model. Let's examine the distributions found by this inference:
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
    ).importance_sample(xs, ys, 10000, 20)
    return plot_posterior(fs, xs, ys)


periodic_ex(quadratic)

# %% [markdown]
# Though the sample points are periodic, we supplied the degree 2 polynomial prior, and got reasonable results quickly. Before trying the Periodic prior, we might try degree 3 polynomials, and see what we get:

# %%
linear = b.Polynomial(max_degree=1, coefficient_d=genjax.normal(0.0, 2.0))
cubic = b.Polynomial(max_degree=3, coefficient_d=genjax.normal(0.0, 2.0))

periodic_ex(cubic)

# %% [markdown]
# **LGTM!**
# Now for periodic prior.
#

# %%
periodic_ex(periodic, key=jax.random.PRNGKey(222))

# %% [markdown]
# Interesting! The posterior is full of good solutions but also contains a sprinkling of Nyquist-impostors!


# %% [markdown]
# ## Conclusion
# I like this framework more than I thought I would when I started it. A wise man once said it is easier to write probabilistic programming languages than probabilistic programs and I found it so. The experience of doing this helped me to understand JAX better. In particular, the idea of creating a custom `Pytree` object seemed exotic to me before I started this, but: if you want to have a Generative Function that produces something other than an array, while retaining full JAX-compatibility, it's exactly what you should do. In this case, the DSL allows the construction and composition of Generative Functions that sample from distributions of real-valued functions on the real line, and that's what curve fitting is about.
#
# ## Grand Finale: the ascending periodic example


# %% [markdown]
# Keeping track of the curve_fit, xs & ys, and other data for different
# experiments we will conduct can be confusing, so we'll write a little
# function that yokes the experiment material into a dict.
# %%
def ascending_periodic_ex(F: b.Block):
    xs = jnp.linspace(-0.9, 0.9, 20)
    ys = (0.7 * xs + 0.3 * jnp.sin(9 * xs + 0.3)).at[7].set(0.75)
    ys += jax.random.normal(key=jax.random.PRNGKey(22), shape=ys.shape) * 0.07
    curve_fit = b.CurveFit(curve=F, sigma_inlier=sigma_inlier, p_outlier=p_outlier)
    fs = curve_fit.importance_sample(xs, ys, 20000, 10)
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
    steps: int = 1,
):
    """Run `n` steps of the update algorithm. `scale` specifies the amount of gaussian drift in SD units."""
    if curve_fit.gf != tr.get_gen_fn():
        raise ValueError("The trace was not generated by the given curve")
    outlier_path = ("ys", ..., "outlier")

    def gaussian_drift_step(key: PRNGKey, tr: genjax.Trace):
        choices = tr.get_choices()

        def update(key: PRNGKey, cm: genjax.ChoiceMap) -> genjax.Trace:
            k1, k2 = jax.random.split(key)
            updated_tr, log_weight, arg_diff, bwd_problem = tr.update(k1, cm)
            log_reject = jnp.minimum(0.0, log_weight)
            mh_choice = jax.random.uniform(key=k2)
            return jax.lax.cond(jnp.log(mh_choice) <= log_reject, lambda: updated_tr, lambda: tr)

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
            flips = jax.random.bernoulli(k1, shape=outlier_states.shape)
            return update(
                k2,
                C["ys", jnp.arange(len(flips)), "outlier"].set(
                    jnp.logical_xor(outlier_states, flips).astype(int)
                ),
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
    sub_keys = jax.random.split(key, (steps, K))
    def t(x):
        return x, x
    final_trace, gaussian_steps = jax.lax.scan(
        lambda trace, keys: t(jax.vmap(gaussian_drift_step)(keys, trace)),
        tr0,
        sub_keys,
    )
    return final_trace, gaussian_steps
# %% [markdown]
# In this cell, we will take the original curve fit size-100 importance sample of
# the degree-2 polynomial prior and run it through 100 steps of Gaussian drift.
# %%
key, sub_key = jax.random.split(jax.random.PRNGKey(314159))
tru, steps = gaussian_drift(sub_key, curve_fit, tr, steps=100, scale=.1)
plot_posterior(tru, xs, ys)
# %% [markdown]
# The Gaussian_drift function returns two values: the final result of the drift, which we plotted above, and the results of each individual Gaussian drift step. We can use the latter value to create an animation, showing Gaussian drift in action.

# %%

def drift_animation(step_trace: genjax.Trace, xs, ys, fps=8):
    max_index = jnp.argmax(step_trace.get_score(), axis=1)
    frames = [plot_posterior(genjax.pytree.nth(step_trace, i), xs, ys,
                             winningIndex=max_index[i])
              for i in range(step_trace.get_score().shape[0])]
    return Plot.Frames(frames, fps=fps)



# %%
drift_animation(steps, xs, ys)

# %% [markdown]
# Now let's try something more difficult. We will gaussian drift the
# periodic example.
# %%

# %%
jnp.argmax(steps.get_score(), axis=0)

# %%
key, sub_key = jax.random.split(key)
periodic_t1, periodic_t1_steps  = gaussian_drift(
    sub_key, periodic_data["curve_fit"], periodic_data["tr"], steps=100
)
drift_animation(periodic_t1_steps, periodic_data["xs"], periodic_data["ys"])
# %% [markdown]
# Let's drift the _result_ of that experiment at a smaller scale and see if that helps
# %%
key, sub_key = jax.random.split(key)
periodic_t2, periodic_t2_steps = gaussian_drift(
    sub_key, periodic_data["curve_fit"], periodic_t1, steps=100, scale=0.005
)
drift_animation(periodic_t2_steps, periodic_data["xs"], periodic_data["ys"])

# %%
pz.ts.render_array(periodic_t2_steps.get_score())
# %%
key, sub_key = jax.random.split(key)

def demo(
    *,
    xs: FloatArray,                                  # x coordinates of points to fit
    ys: FloatArray,                                  # y coordinates of points to fit
    noise_scale: float,                              # SD of noise added to ys
    gaussian_drift_scale: float,                     # scale of Gaussian drift proposals
    gaussian_drift_steps: int,                       # number of drift steps to take
    sigma_inlier: genjax.GenerativeFunctionClosure,  # distribution of inlier sigma
    p_outlier: genjax.GenerativeFunctionClosure,     # distribution of p_outlier
    prior: b.Block,                                  # function shape
    K: int,                                          # number of particles
    N: int,                                          # size of importance sample per particle
    fps: int,                                        # fps hint for generated animation
    key: PRNGKey):                                   # seed
    k1, k2, k3 = jax.random.split(key, 3)
    ys += jax.random.normal(k1, shape=ys.shape) * noise_scale
    curve_fit = b.CurveFit(curve=prior, sigma_inlier=sigma_inlier, p_outlier=p_outlier)
    tr = curve_fit.importance_sample(xs, ys, N, K, key=k2)
    _best, steps = gaussian_drift(k3, curve_fit, tr, gaussian_drift_scale, gaussian_drift_steps)
    return drift_animation(steps, xs, ys, fps=fps)


xs=jnp.linspace(-0.9, 0.9, 20)

demo(
    key=sub_key,
    xs=xs,
    ys=xs ** 2.0 - 0.5,
    noise_scale=0.1,
    gaussian_drift_scale=0.05,
    gaussian_drift_steps=24,
    sigma_inlier=genjax.uniform(0.0, 0.3),
    p_outlier=genjax.beta(1.0, 1.0),
    prior=quadratic,
    K=10,
    N=2000,
    fps=2,
)
# %%
