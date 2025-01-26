export class FPSCounter {
  private readonly frameTimes: number[] = []

  public observe(): number {
    const now = performance.now()
    this.frameTimes.push(now)
    while (this.frameTimes.length && this.frameTimes[0] < now - 1000)
      this.frameTimes.shift()
    return this.frameTimes.length
  }

  public reset() {
    this.frameTimes.length = 0
  }
}
