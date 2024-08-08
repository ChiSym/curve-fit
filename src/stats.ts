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

  summarize(): NormalParams {
    return {
      mu: this.mean,
      sigma: Math.sqrt(this.m2 / (this.count - 1)),
    }
  }

  reset(): void {
    this.count = 0
    this.mean = 0
    this.m2 = 0
  }
}
