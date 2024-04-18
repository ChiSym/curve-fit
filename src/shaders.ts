export const pcg3d = /*glsl*/ `
uvec3 pcg3d(uvec3 v) {
  // Citation: Mark Jarzynski and Marc Olano, Hash Functions for GPU Rendering,
  // Journal of Computer Graphics Techniques (JCGT), vol. 9, no. 3, 21-38, 2020
  // Available online http://jcgt.org/published/0009/03/02/

  v = v * 1664525u + 1013904223u;
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  v ^= v >> 16u;
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  return v;
}

void split(inout uvec3 key, out uvec3 sub_key) {
  sub_key = pcg3d(key);
  key = pcg3d(sub_key);
}
`

export const random_uniform = /*glsl*/ `
// recovered from de-compiled JAX
float random_uniform(uvec3 seed, float low, float high) {
  float a = uintBitsToFloat(seed.x >> 9u | 1065353216u) - 1.0;
  float diff = high - low;
  float w = diff * a;
  float u = w + low;
  return max(low, u);
}
`

export const logpdf_uniform = /*glsl*/ `
// recovered from de-compiled JAX
float logpdf_uniform(float v, float low, float high) {
  bool d = v != v;
  bool e = v < low;
  bool f = v > high;
  // g = e, h = f
  bool i = e || f;
  float j = high - low;
  float k = 1.0 / j;
  float l = i ? 0.0 : k;
  float q = d ? v : l;
  return log(q);
}
`

export const flip = /*glsl*/ `
bool flip(uvec3 seed, float prob) {
  if (prob >= 1.0) return true;
  float a = random_uniform(seed, 0.0, 1.0);
  return a < prob;
}
`

export const erfc = /*glsl*/ `
// From Press NR 3ed.
// A lower-order Chebyshev approximation produces a very concise routine, though with only about single precision accuracy:
// Returns the complementary error function with fractional error everywhere less than 1.2e-7.
float erfc(float x) {
  float t,z=abs(x),ans;
  t=2./(2.+z); ans=t*exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
    t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
    t*(-0.82215223+t*0.17087277)))))))));
  return (x >= 0.0 ? ans : 2.0-ans);
}
`

export const inv_erfc = /*glsl*/ `
// The following two functions are from
// http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function/
float inv_erfc(float x) {
  float pp, t, r, er;

  if(x < 1.0) {
    pp = x;
  } else {
    pp = 2.0 - x;
  }
  t = sqrt(-2.0 * log(pp/2.0));
  r = -0.70711 * ((2.30753 + t * 0.27061)/(1.0 + t * (0.99229 + t * 0.04481)) - t);
  er = erfc(r) - pp;
  r += er/(1.12837916709551257 * exp(-r * r) - r * er);
  //Comment the next two lines if you only wish to do a single refinement
  //err = erfc(r) - pp;
  //r += err/(1.12837916709551257 * exp(-r * r) - r * er);
  if(x > 1.0) {
    r = -r;
  }
  return r;
}
`

export const inv_erf = /*glsl*/ `
float inv_erf(float x){
  return inv_erfc(1.0-x);
}
`

export const random_normal = /*glsl*/ `
float random_normal(uvec3 seed, float loc, float scale) {
  float u = sqrt(2.0) * inv_erf(random_uniform(seed, -1.0, 1.0));
  return loc + scale * u;
}
`

export const logpdf_normal = /*glsl*/ `
// Decompiled from JAX genjax.normal.logpdf
float logpdf_normal(float v, float loc, float scale) {
  float d = v / scale;
  float e = loc / scale;
  float f = d - e;
  float g = pow(f, 2.0);
  float h = -0.5 * g;
  float i = log(scale);
  float k = 0.9189385175704956 + i;
  return h - k;
}
`

export const compute_shader = /*glsl*/ `#version 300 es
  ${pcg3d}
  ${random_uniform}
  ${logpdf_uniform}
  ${flip}
  ${erfc}
  ${inv_erfc}
  ${inv_erf}
  ${random_normal}
  ${logpdf_normal}

  #define N_POINTS 10
  #define N_POLY 3
  #define N_SAMPLES 50

  uniform float xs[N_POINTS];
  uniform float ys[N_POINTS];

  in float a;
  out vec3 model;
  flat out uint outliers;
  out float weight;

  vec3 sample_alpha(uvec3 key) {
    uvec3 sub_key;
    vec3 alpha;
    split(key, sub_key);
    alpha[0] = random_normal(sub_key, 0.0, 2.0);
    split(key, sub_key);
    alpha[1] = random_normal(sub_key, 0.0, 2.0);
    split(key, sub_key);
    alpha[2] = random_normal(sub_key, 0.0, 2.0);
    return alpha;
  }

  float evaluate_poly(vec3 coefficients, float x) {
    return coefficients[0] + x * coefficients[1] + x * x * coefficients[2];
  }

  struct result {
    uint outliers;
    vec3 model;
    float weight;
  };

  result curve_fit_importance(uvec3 key) {
    // Find the importance of the model generated from
    // coefficients. The "choicemap" in this case is one
    // that sets the ys to the observed values. The model
    // has an outlier probability, two normal ranges for
    // inlier and outlier, and a fixed set of xs. We generate
    // the y values from the polynomial, and compute the
    // logpdf of these given the expected values and the
    // outlier choices. Sum all that up and it's the score of
    // the model.
    float w = 0.0;
    uint outliers = 0u;
    uvec3 sub_key;
    split(key, sub_key);
    vec3 coefficients = sample_alpha(sub_key);
    for (int i = 0; i < N_POINTS; ++i) {
      split(key, sub_key);
      bool outlier = flip(sub_key, 0.1);
      outliers = outliers | (uint(outlier) << i);
      float y_model = evaluate_poly(coefficients, xs[i]);
      float y_observed = ys[i];
      w += logpdf_normal(y_observed, y_model, outlier ? 30.0 : 0.3);
    }
    return result(outliers, coefficients, w);
  }


  void main() {
    uvec3 key = pcg3d(uvec3(uint(a), ~uint(a), 877)), sub_key;
    split(key, sub_key);
    result r = curve_fit_importance(sub_key);
    outliers = r.outliers;
    weight = r.weight;
    model = r.model;
  }
`

export const render_shader = /*glsl*/ `#version 300 es
precision highp float;
#define N_POINTS 10
uniform vec2 canvas_size;
//uniform vec2 points[N_POINTS];
out vec4 out_color;
//uniform vec3 models[];

void main() {
  // Map pixel coordinates [0,w) x [0,h) to the unit square [-1, 1) x [-1, 1)
  vec2 xy = gl_FragCoord.xy / canvas_size.xy * 2.0 + vec2(-1.0,-1.0);
  vec2 o = vec2(0.0, 0.0);
  float d = distance(xy,o);
  // might need a smoothstep in here to antialias
  if (d < 0.05) {
    out_color = vec4(1.0,0.0,0.0,1.0);
  } else {
    out_color = vec4(0.8,0.8,0.8,1.0);
  }
}
`
