# %%
import jax.numpy as jnp
import jax.random
import genjax
from genjax import ChoiceMapBuilder as C
import genjax_blocks as b
# %%
coef_d = genjax.normal(0.0, 2.0)
sigma_inlier = genjax.uniform(0.0, 0.3)
quadratic = b.Polynomial(max_degree=2, coefficient_d=coef_d)
p_outlier = genjax.beta(1.0, 1.0)
xs = jnp.arange(4.)
ys = jnp.array([.1,.3,.9,.15])
cf = b.CurveFit(curve=quadratic, sigma_inlier=sigma_inlier, p_outlier=p_outlier)
# %%
def logpdf_normal(v, loc, scale):
    d = v / scale
    e = loc / scale
    f = d - e
    g = f ** 2
    h = -0.5 * g
    i = jnp.log(scale)
    k = 0.9189385175704956 + i
    return h - k

def logpdf_flip(v, p):
    g = -p
    h = jnp.log1p(g)
    i = jnp.log(p)
    k = 1.0 - v
    l = k == 0.0
    n = h * k
    o = jax.lax.select_n(l, n, 0.0)
    q = v == 0
    r = i * v
    s = jax.lax.select_n(q, r, 0.0)
    return o + s

def logpdf_uniform(v, low, high):
    d = v != v
    e = v < low
    f = v > high
    i = jnp.logical_or(e, f)
    j = high - low
    k = 1.0 / j
    l = jax.lax.select(i, 0.0, k)
    q = jax.lax.select(d, v, l)
    return jnp.log(q)

def importance_sample(cf, N, key):
    # categorically sample 1 of N. This is used to avoid drawing curves with score == -inf
    k1, k2 = jax.random.split(key)
    cmap = C['ys', jnp.arange(4), 'y', 'value'].set(ys)
    trs, ws = jax.vmap(cf.jitted_importance, in_axes=(0, None, None))(jax.random.split(k1, N), cmap, (xs,))
    winner = cf.categorical_sampler(k2, ws)
    return jax.tree.map(lambda v: v[winner], trs), ws[winner]


def importance_score(tr):
    # TODO: get ys out of trace, use "our" flip
    choices = tr.get_choices()
    sigma_in = choices['sigma_inlier']
    fn = tr.subtraces[3].inner.get_args()[1]
    tr_ys = fn(xs)[:,0]
    #print('tr_ys', tr_ys)
    #tr_p0 = choices['p_outlier']
    outliers = choices['ys',...,'outlier']
    #print('outliers', outliers, 'tr_p0', tr_p0)
    def y_score(y, outlier, tr_y):
        return jax.lax.select(
            outlier,
            logpdf_uniform(y, -1.0, 1.0),
            logpdf_normal(tr_y, y, sigma_in)
        )
    tr_y_logpdfs = jax.vmap(y_score)(ys, outliers, tr_ys)
    #print('tr_y_logpdfs', tr_y_logpdfs)
    return jnp.sum(tr_y_logpdfs)



# %%
def test_trace_importance(key):
    tr, w = importance_sample(cf, 1000, key)
    #print([t.get_score() for t in tr.subtraces])
    #print('genjax reference', w)
    shader = importance_score(tr)
    #print('shader method', shader)
    return shader - w

# %%
tti = jax.jit(test_trace_importance)
jnp.sum(jnp.abs(jax.vmap(tti)(jax.random.split(jax.random.PRNGKey(314159), 100))))

# %%
def update_coefficient(tr: genjax.Trace, *, scale=0.01, key):
    # get existing coefficients
    k1, k2 = jax.random.split(key)
    c0 = tr.subtraces[0].subtraces[0].get_retval()
    # perturb them
    delta = scale * jax.random.normal(k1, c0.shape)
    c1 = c0 + delta

    # See what happens when we set the constant term of the coefficients to 1.
    cm = C['curve', 'p', jnp.arange(c0.shape[0]), 'coefficients'].set(c1)
    tr_u, w_u, _, _ = tr.update(key, cm)

    # Do the score outside jax
    c_loc, c_scale = coef_d.args
    c0s = jnp.array([logpdf_normal(c, c_loc, c_scale) for c in c0])
    c1s = jnp.array([logpdf_normal(c, c_loc, c_scale) for c in c1])
    # print(f'delta: {delta}')
    # print(f'c0   : {c0}')
    # print(f'c1   : {c1}')
    # print(f'c0s  : {c0s} {jnp.sum(c0s)}')
    # print(f'c1s  : {c1s} {jnp.sum(c1s)}')
    # print(f'c1-0s: {c1s - c0s}')
    # print(f'diff : {jnp.sum(c0s) - jnp.sum(c1s)}')
    choices = tr.get_choices()
    sigma_in = choices['sigma_inlier']
    outliers = choices['ys',...,'outlier']
    # print(f'outs : {outliers}')

    # now do the ys
    def y_score(y, outlier, tr_y, new_y):
        return jax.lax.select(
            outlier,
            0.0, # the target ys don't change, and that's what this part of the score is based
            logpdf_normal(new_y, y, sigma_in) - logpdf_normal(tr_y, y, sigma_in)
        )
    fn = tr.subtraces[3].inner.get_args()[1]
    tr_ys = fn(xs)[:,0]
    new_fn = tr_u.subtraces[3].inner.get_args()[1]
    new_ys = new_fn(xs)[:, 0]
    y_scores = jax.vmap(y_score)(ys, outliers, tr_ys, new_ys)
    # print(f'y_difs: {y_scores}')

    #imp = importance_score(tr_u) - importance_score(tr)
    imp = jnp.sum(y_scores)
    # print(f'importance score {imp}')

    # I think this accounts for the difference in the 0th subtrace. There's a little gap to fill though.
    our_w  = jnp.sum(c1s - c0s) + imp
    # print(f'our_w: {our_w}')
    return tr_u, w_u, our_w

def update_score(tr, cm):
    pass

def test_update(key):
    k1, k2 = jax.random.split(key)
    #tr, w = importance_sample(cf, 1000, k1)
    cmap = C['ys', jnp.arange(4), 'y', 'value'].set(ys)
    tr, w = cf.gf.importance(k1, cmap, (xs,))
    #tr, w = importance_sample(cf, 1, k1)
    tr_u, w_u, our_w = update_coefficient(tr, key=k2)
    return tr, tr_u, w_u, our_w
# %%
tr0, tr_u, w_u, our_w = test_update(jax.random.PRNGKey(113))
w_u, our_w, w_u - our_w
# %%
def test_update_batch(k):
    _, _, w_u, our_w = test_update(k)
    return w_u - our_w

jax.vmap(test_update_batch)(jax.random.split(jax.random.PRNGKey(314159) ,100))
# %%
# Bug?
k = jax.random.PRNGKey(271828)
tr_unjitted, _ = cf.gf.importance(k, C.n(), (xs,))
tr_jitted, _ = jax.jit(cf.gf.importance)(k, C.n(), (xs,))
# %%
tr_unjitted.subtraces[2], tr_jitted.subtraces[2]

# %%
@genjax.gen
def repro_model():
    return genjax.beta(1.0, 1.0) @ 'x'

k = jax.random.PRNGKey(271828)
m_unjitted, _ = repro_model.importance(k, C.n(), ())
m_jitted, _ = jax.jit(repro_model.importance)(k, C.n(), ())
m_unjitted.get_score(), m_jitted.get_score()

# %%
m_jitted
# %%
genjax.beta.logpdf(0.6, 1.0, 1.0), jax.jit(genjax.beta.logpdf)(0.6, 1.0, 1.0)

# %%
s_unjitted = repro_model.simulate(k, ())
s_jitted = jax.jit(repro_model.simulate)(k, ())
# %%
s_unjitted.get_score(), s_jitted.get_score()

# %%
