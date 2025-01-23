fn pcg3d(v: vec3u) -> vec3u {
  // Citation: Mark Jarzynski and Marc Olano, Hash Functions for GPU Rendering,
  // Journal of Computer Graphics Techniques (JCGT), vol. 9, no. 3, 21-38, 2020
  // Available online http://jcgt.org/published/0009/03/02/

  var w: vec3u = v * 1664525u + 1013904223u;
  w.x += w.y*w.z; w.y += w.z*w.x; w.z += w.x*w.y;
  w ^= w >> vec3u(16, 16, 16);
  w.x += w.y*w.z; w.y += w.z*w.x; w.z += w.x*w.y;
  return w;
}

// From Press NR 3ed.
// A lower-order Chebyshev approximation produces a very concise routine, though with only about single precision accuracy:
// Returns the complementary error function with fractional error everywhere less than 1.2e-7.
fn erfc(x: f32) -> f32 {
  var z = abs(x);
  var t=2./(2.+z);
  var ans=t*exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
    t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
    t*(-0.82215223+t*0.17087277)))))))));
  return select(2.0 - ans, ans, x >= 0.0);
}

// The following two functions are from
// http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function/
fn inv_erfc(x: f32) -> f32 {
  let pp: f32 = select(2.0 - x, x, x < 1.0);
  let t: f32 = sqrt(-2.0 * log(pp/2.0));
  var r: f32;
  var er: f32;

  r = -0.70711 * ((2.30753 + t * 0.27061)/(1.0 + t * (0.99229 + t * 0.04481)) - t);
  er = erfc(r) - pp;
  r += er/(1.12837916709551257 * exp(-r * r) - r * er);
  //Comment the next two lines if you only wish to do a single refinement
  //err = erfc(r) - pp;
  //r += err/(1.12837916709551257 * exp(-r * r) - r * er);
  r = select(r, -r, x>1.0);
  return r;
}

fn inv_erf(x: f32) -> f32 {
  return inv_erfc(1.0-x);
}

fn random_normal(loc: f32, scale: f32) -> f32 {
  let u = sqrt(2.0) * inv_erf(random_uniform(-1.0, 1.0));
  return loc + scale * u;
}

// recovered from de-compiled JAX
fn random_uniform(low: f32, high: f32) -> f32 {
  seed = pcg3d(seed);
  let a: f32 = bitcast<f32>((seed.x >> 9u) | 1065353216u) - 1.0;
  let diff = high - low;
  let w = diff * a;
  let u = w + low;
  return max(low, u);
}
