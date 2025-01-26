export interface NormalParams {
  mu: number
  sigma: number
}

export class RunningStats {
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
  private count: number
  private mean: number
  private m2: number

  constructor() {
    this.count = 0
    this.mean = 0
    this.m2 = 0
  }

  observe(value: number): void {
    this.count += 1
    const delta = value - this.mean
    this.mean += delta / this.count
    const delta2 = value - this.mean
    this.m2 += delta * delta2
  }

  summarize(): XDistribution {
    return Normal(this.mean, Math.sqrt(this.m2 / (this.count - 1)))
  }

  reset(): void {
    this.count = 0
    this.mean = 0
    this.m2 = 0
  }
}

export class XDistribution {
  public readonly shape: DistributionShape
  public readonly parameters: Map<string, number>

  constructor(shape: DistributionShape, parameters: number[]) {
    this.shape = shape
    if (parameters.length != this.shape.parameterName.length) {
      throw new Error(
        `incorrect number of paramters for ${this.shape.name} distribution: ${parameters.length} != ${this.shape.parameterName.length}`,
      )
    }
    this.parameters = new Map()
    for (let i = 0; i < parameters.length; ++i) {
      this.parameters.set(this.shape.parameterName[i], parameters[i])
    }
  }
  public get(pName: string): number {
    return this.parameters.get(pName)!
  }
  public set(pName: string, value: number) {
    return this.parameters.set(pName, value)
  }
  public assoc(pName: string, value: number) {
    const ret = this.clone()
    ret.set(pName, value)
    return ret
  }
  public clone(): XDistribution {
    return new XDistribution(this.shape, Array.from(this.parameters.values()))
  }
  public getParameterNames(): string[] {
    return this.shape.parameterName
  }
}

class DistributionShape {
  public readonly name: string
  public readonly parameterName: string[]
  constructor(name: string, parameterName: string[]) {
    this.name = name
    this.parameterName = parameterName
  }
}

const NormalDistributionShape = new DistributionShape("normal", ["mu", "sigma"])

export const Normal = (mu: number, sigma: number) =>
  new XDistribution(NormalDistributionShape, [mu, sigma])

export function logpdf_normal(v: number, loc: number, scale: number) {
  const d = v / scale
  const e = loc / scale
  const f = d - e
  const g = f ** 2
  const h = -0.5 * g
  const i = Math.log(scale)
  const k = 0.9189385175704956 + i
  return h - k
}

export function logpdf_flip(v: number, p: number) {
  const g = -p
  const h = Math.log1p(g)
  const i = Math.log(p)
  const k = 1.0 - v
  const l = k == 0.0
  const n = h * k
  const o = l ? 0 : n
  const q = v == 0
  const r = i * v
  const s = q ? 0 : r
  return o + s
}

export function logpdf_uniform(v: number, low: number, high: number) {
  const d = v != v
  const e = v < low
  const f = v > high
  const i = e || f
  const j = high - low
  const k = 1.0 / j
  const l = i ? 0 : k
  const q = d ? v : l
  return Math.log(q)
}

export function sample_normal() {
  const u1 = 1 - Math.random()
  const u2 = Math.random()
  const mag = Math.sqrt(-2 * Math.log(u1))
  return mag * Math.cos(2 * Math.PI * u2)
}
