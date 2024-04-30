export interface Normal {
  mu: number
  sigma: number
}

export interface Model {
  model: Float32Array
  outliers: number
  weight: number
  params: Float32Array
}

export const MODEL_SIZE = 3
