
export type Normal = {
  mu: number,
  sigma: number
}

export type Model = {
  model: Float32Array,
  outliers: number,
  weight: number,
  params: Float32Array,
}

export const MODEL_SIZE = 3
