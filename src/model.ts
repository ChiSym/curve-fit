export interface Normal {
  mu: number
  sigma: number
}

export interface Model {
  model: Float32Array
  outlier: number
  log_weight: number
  p_outlier: number
}
