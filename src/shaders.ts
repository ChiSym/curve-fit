const pcg3d = /* glsl */ `

// The rules: if you use any part of the seed, it's your responsibility
// to call this routine to mutate the seed for the next user.

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
`

const lgamma = /* glsl */ `

const float lgamma_coefficient[] = float[6](
  76.18009172947146,
  -86.50532032941677,
  24.01409824083091,
  -1.231739572450155,
  0.1208650973866179e-2,
  -0.5395239384953e-5
);

float lgamma(float xx) {
  // Numerical Recipes in C++ 2ed. p.219
  float y = xx, x = xx, tmp = xx + 5.5;
  tmp -= (x+0.5)*log(tmp);
  float ser =  1.000000000190015;
  for (int j = 0; j < 6; ++j) ser += lgamma_coefficient[j]/++y;
  return -tmp+log(2.5066282746310005*ser/x);
}

`

const randomUniform = /* glsl */ `
// recovered from de-compiled JAX
float random_uniform(inout uvec3 seed, float low, float high) {
  float a = uintBitsToFloat(seed.x >> 9u | 1065353216u) - 1.0;
  seed = pcg3d(seed);
  float diff = high - low;
  float w = diff * a;
  float u = w + low;
  return max(low, u);
}
`

const logpdfUniform = /* glsl */ `
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

const logpdfFlip = /* glsl */ `
// recovered from de-compiled JAX
float logpdf_flip(float v, float p) {
  float g = -p;
  float h = log(g + 1.0);  // log1p
  float i = log(p);
  float k = 1.0 - v;
  bool l = k == 0.0;
  float n = h * k;
  float o = l ? 0.0 : n;
  bool q = i == 0.0;
  float r = i * v;
  float s = q ? 0.0 : r;
  return o + s;
}
`

const flip = /* glsl */ `
bool flip(inout uvec3 seed, float prob) {
  if (prob >= 1.0) return true;
  float a = random_uniform(seed, 0.0, 1.0);
  return a < prob;
}
`

const erfc = /* glsl */ `
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

const invErfc = /* glsl */ `
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

const invErf = /* glsl */ `
float inv_erf(float x){
  return inv_erfc(1.0-x);
}
`

const randomNormal = /* glsl */ `
float random_normal(inout uvec3 seed, float loc, float scale) {
  float u = sqrt(2.0) * inv_erf(random_uniform(seed, -1.0, 1.0));
  return loc + scale * u;
}
`

const logpdfNormal = /* glsl */ `
// De-compiled from JAX genjax.normal.logpdf
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

const stdlib = `
  ${pcg3d}
  ${lgamma}
  ${randomUniform}
  ${logpdfUniform}
  ${logpdfFlip}
  ${flip}
  ${erfc}
  ${invErfc}
  ${invErf}
  ${randomNormal}
  ${logpdfNormal}
`

export function importanceShader(nParameters: number): string {
  return /* glsl */ `#version 300 es

  ${stdlib}

  #define N_POINTS 10u
  #define N_POLY 3
  #define N_SAMPLES 50
  #define M_PI 3.1415926535897932384626433832795

  uniform vec2 points[N_POINTS];
  uniform float alpha_loc[${nParameters}];
  uniform float alpha_scale[${nParameters}];
  uniform uint component_enable;

  in uvec3 seed;

  out float out_0;
  out float out_1;
  out float out_2;
  out float out_3;
  out float out_4;
  out float out_5;
  out float out_6;
  out float out_log_weight;
  out float out_p_outlier;
  out float out_outliers;
  out float out_inlier_sigma;

  vec3 sample_poly(inout uvec3 seed) {
    if ((component_enable & 1u) != 0u) {
      return vec3(
        random_normal(seed, alpha_loc[0], alpha_scale[0]),
        random_normal(seed, alpha_loc[1], alpha_scale[1]),
        random_normal(seed, alpha_loc[2], alpha_scale[2])
      );
    } else {
      return vec3(0.0, 0.0, 0.0);
    }
  }

  vec3 sample_periodic(inout uvec3 seed) {
    if ((component_enable & 2u) != 0u) {
      return vec3(
        random_normal(seed, alpha_loc[3], alpha_scale[3]),
        random_normal(seed, alpha_loc[4], alpha_scale[4]),
        random_normal(seed, alpha_loc[5], alpha_scale[5])
      );
    } else {
      return vec3(0.0, 0.0, 0.0);
    }
  }

  float evaluate_poly(vec3 coefficients, float x) {
    // components x, y, z map to a_0, a_1, a_2
    return coefficients.x + x * coefficients.y + x * x * coefficients.z;
  }

  float evaluate_periodic(vec3 parameters, float x) {
    // components x, y, z map to omega, A, phi
    return parameters.y * sin(parameters.z + parameters.x * x);
  }

  float evaluate_model(in vec3 polynomial_parameters, in vec3 periodic_parameters, in float x) {
    float y_model = 0.0;
    if ((component_enable & 1u) != 0u) {
      y_model += evaluate_poly(polynomial_parameters, x);
    }
    if ((component_enable & 2u) != 0u) {
      y_model += evaluate_periodic(periodic_parameters, x);
    }
    return y_model;
  }

  void curve_fit_importance(inout uvec3 seed) {
    // Find the importance of the model generated from
    // coefficients. The "choice map" in this case is one
    // that sets the ys to the observed values. The model
    // has an outlier probability, two normal ranges for
    // inlier and outlier, and a fixed set of xs. We generate
    // the y values from the curve, and compute the
    // logpdf of these given the expected values and the
    // outlier choices. Sum all that up and it's the score of
    // the model.
    float inlier_sigma = max(1e-6, random_normal(seed, alpha_loc[6], alpha_scale[6]));
    //float inlier_sigma = 0.3;
    float log_w = 0.0;
    uint outlier_bits = 0u;
    vec3 polynomial_parameters = sample_poly(seed);
    vec3 periodic_parameters = sample_periodic(seed); // how to update seed?
    float p_outlier = random_uniform(seed, 0.0, 1.0);
    for (uint i = 0u; i < N_POINTS; ++i) {
      bool outlier = flip(seed, p_outlier);
      outlier_bits = outlier_bits | (uint(outlier) << i);
      float y_model = evaluate_model(polynomial_parameters, periodic_parameters, points[i].x);
      if (outlier) {
        log_w += logpdf_uniform(y_model, -1.0, 1.0);
      } else {
        log_w += logpdf_normal(points[i].y, y_model, inlier_sigma);
      }
    }
    log_w += logpdf_uniform(p_outlier, 0.0, 1.0);  // TODO: #define
    out_0 = polynomial_parameters[0];
    out_1 = polynomial_parameters[1];
    out_2 = polynomial_parameters[2];
    out_3 = periodic_parameters[0];
    out_4 = periodic_parameters[1];
    out_5 = periodic_parameters[2];
    out_6 = inlier_sigma;
    out_log_weight = log_w;
    out_p_outlier = p_outlier;
    out_outliers = float(outlier_bits);
    out_inlier_sigma = inlier_sigma;
  }


  void main() {
    uvec3 seed = pcg3d(seed);
    curve_fit_importance(seed);
  }
`
}

// // we're not using this yet; instead we're going to prototype it in JS and
// // move it here when we figure out what we're doing
// const curveFitDrift = /* glsl */ `

//   ${stdlib}

//   in uvec3 seed;
//   in vec3 poly_parameters;
//   in vec3 periodic_parameters;

//   void curve_fit_drift(inout uvec3 seed) {
//     // Our task: drift the components of the model, recompute the y,
//     // evaluate the ratio of the old to the new, "flip coin" to accept/reject,
//     // return the data.
//     const float inlier_sigma = 0.3; // FIXME
//     vec3 drift_poly = sample_poly(seed);
//     vec3 drift_periodic = sample_periodic(seed);
//     for (uint i = 0u; i < N_POINTS; ++i) {
//       float x = points[i].x;
//       float y_orig = evaluate_model(orig_poly, orig_periodic, x);
//       float y_drift = evaluate_model(drift_poly, drift_periodic, x);
//       float w_orig = logpdf_normal(points[i].y, y_orig, outlier ? 3.0 : inlier_sigma);
//       float w_drift = logpdf_normal(points[i].y, y_drift, outlier ? 3.0 : inlier_sigma);
//       float w_score = y_drift / y_orig;
//       float u = random_uniform(seed, 0.0, 1.0);
//       if (u <= exp(w_score)) {
//         // accept
//         out_0 = drift_poly[0];
//         out_1 = drift_poly[1];
//         out_2 = drift_poly[2];
//         out_3 = drift_periodic[0];
//         out_4 = drift_periodic[1];
//         out_5 = drift_periodic[2];
//         out_accept = 1;
//         out_log_weight = w_drift;
//       } else {
//         out_0 = orig_poly[0];
//         out_1 = orig_poly[1];
//         out_2 = orig_poly[2];
//         out_3 = orig_periodic[0];
//         out_4 = orig_periodic[1];
//         out_5 = orig_periodic[2];
//         out_accept = 0;
//         out_log_weight = w_orig;
//       }
//     }
//   }
// `

export const computeFragmentShader = /* glsl */ `#version 300 es
  precision highp float;
  void main() {
  }
`
