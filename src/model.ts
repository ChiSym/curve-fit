import {
  logpdf_normal,
  logpdf_uniform,
  sample_normal,
  XDistribution,
} from "./stats"
import { TypedObject } from "./utils"

export class Model {
  constructor(
    coefficients: Float32Array,
    outlier: number,
    log_weight: number,
    p_outlier: number,
    inlier_sigma: number,
  ) {
    this.model = coefficients
    this.outlier = outlier
    this.log_weight = log_weight
    this.p_outlier = p_outlier
    this.inlier_sigma = inlier_sigma
  }

  public readonly model: Float32Array
  public readonly outlier: number
  public readonly log_weight: number
  public readonly p_outlier: number
  public inlier_sigma: number
  public fn(x: number) {
    return Model.fn_from_coefficients(this.model, x)
  }
  static fn_from_coefficients(c: Float32Array, x: number) {
    const y_poly = c[0] + x * c[1] + x * x * c[2]
    const y_periodic = c[4] + Math.sin(c[5] + c[3] * x)
    return y_poly + y_periodic
  }

  public drift_coefficients(
    scale: number,
    coefficients: TypedObject<XDistribution>,
    points: number[][],
  ) {
    const drifted = this.model.map((v) => v + scale * sample_normal())
    const old_ys = points.map(([x]) => this.fn(x))
    const new_ys = points.map(([x]) => Model.fn_from_coefficients(drifted, x))
    // compute update score
    // hm. we need the distribution that generates coefficients, which
    // we don't have here. Maybe this function belongs somewhere else?
    let log_w_cs = 0.0
    let log_w_ys = 0.0

    Object.values(coefficients).forEach((dist, i) => {
      if (i < 6) {
        // TODO: bit of a hack to avoid perturbing sigma_inlier here
        const mu = dist.get("mu")
        const sigma = dist.get("sigma")
        log_w_cs +=
          logpdf_normal(drifted[i], mu, sigma) -
          logpdf_normal(this.model[i], mu, sigma)
      }
    })

    const sigma_inlier = coefficients.inlier.get("mu")
    for (let i = 0; i < points.length; ++i) {
      const inlier = (this.outlier & (1 << i)) == 0
      const y_i = points[i][1]
      if (inlier) {
        log_w_ys +=
          logpdf_normal(new_ys[i], y_i, sigma_inlier) -
          logpdf_normal(old_ys[i], y_i, sigma_inlier)
      }
    }
    console.log(`cs ${log_w_cs} ys ${log_w_ys}`)
    const log_w = log_w_cs + log_w_ys
    const choice = Math.random()
    if (choice <= Math.exp(log_w)) {
      // accept
      console.log(`${choice} ${Math.exp(log_w)} accepted`)
      this.model.set(drifted)
    } else {
      console.log(`${choice} ${Math.exp(log_w)} rejected`)
    }
  }

  public drift_sigma_inlier(
    scale: number,
    coefficients: TypedObject<XDistribution>,
    points: number[][],
  ) {
    const old_inlier_sigma = coefficients.inlier.get("mu")
    const sigma_sigma = coefficients.inlier.get("sigma")
    const delta = scale * sample_normal()
    const new_inlier_sigma = old_inlier_sigma + delta
    let log_w = logpdf_uniform(new_inlier_sigma, old_inlier_sigma, sigma_sigma)
    // now compute the likelihoods of the inlier y's under this sigma
    const ys = points.map(([x]) => this.fn(x))
    for (let i = 0; i < ys.length; ++i) {
      if ((this.outlier & (1 << i)) == 0) {
        // inlier
        log_w += logpdf_normal(ys[i], points[i][1], new_inlier_sigma)
      }
    }

    const choice = Math.random()
    if (choice <= Math.exp(log_w)) {
      // accept
      console.log(
        `accepted inlier_sigma update ${this.inlier_sigma} -> ${new_inlier_sigma}`,
      )
      this.inlier_sigma = new_inlier_sigma
    }
  }
}

// private drift(models: Model[], mParams: ModelParameters, driftScale: number) {
//   // prototype of drift in JS before shader implementation

//   function evaluate(coefficients: Float32Array, x: number) {
//     const y_poly =
//       coefficients[0] + x * coefficients[1] + x * x * coefficients[2]
//     const y_periodic =
//       coefficients[4] + Math.sin(coefficients[5] + coefficients[3] * x)
//     return y_poly + y_periodic
//   }

//   const xs = mParams.points.map((p) => p[0])

//   for (const m of models) {
//     const parameters = m.model
//     const drifted = parameters.map((v) => v + driftScale * random_normal())
//     const drifted_ys = xs.map((x) => evaluate(drifted, x))
//   }
// }
