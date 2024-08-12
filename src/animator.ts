import { XDistribution } from "./App.tsx"
import { GPGPU_Inference, InferenceParameters } from "./gpgpu.ts"
import { Render } from "./render.ts"
import { RunningStats } from "./stats.ts"

function log(level: string, message: unknown): void {
  if (level === "error") {
    console.error(message)
  } else {
    console.info(message)
  }
  const d = document.createElement("div")
  d.className = "log-" + level
  const t = document.createTextNode(message + "")
  d.appendChild(t)
  document.querySelector("#app")?.appendChild(d)
}

export interface InferenceReport {
  fps: number
  ips: number
  totalFailedSamples: number
  pOutlierStats: RunningStats
  autoSIR: boolean
}

export class Animator {
  private modelParameters: Map<string, XDistribution>
  private readonly inferenceParameters: InferenceParameters
  private readonly inferenceReportCallback: (r: InferenceReport) => void
  private readonly stats: Map<string, RunningStats>
  private points: number[][] = []
  private pause: boolean = false
  private autoSIR: boolean = false
  private componentEnable: Map<string, boolean> = new Map()
  private frameCount = 0
  private totalFailedSamples = 0
  private t0: DOMHighResTimeStamp = performance.now()

  // TODO: define type ModelParameters as Map<string, Distribution>; change signature of inference
  // engine code to take multiple parameters, giving up old name

  constructor(
    modelParameters: Map<string, XDistribution>,
    inferenceParameters: InferenceParameters,
    inferenceReportCallback: (r: InferenceReport) => void,
  ) {
    this.inferenceReportCallback = inferenceReportCallback
    // make copies of the initial values
    this.modelParameters = new Map(modelParameters.entries())
    this.inferenceParameters = Object.assign({}, inferenceParameters)
    this.stats = new Map(
      Array.from(modelParameters.keys()).map((k) => [k, new RunningStats()]),
    )
  }

  public setInferenceParameters(ps: InferenceParameters) {
    this.inferenceParameters.importanceSamplesPerParticle =
      ps.importanceSamplesPerParticle
    this.inferenceParameters.numParticles = ps.numParticles
  }

  public setModelParameters(params: Map<string, XDistribution>) {
    this.modelParameters = new Map(params)
    this.stats.forEach((s) => s.reset())
  }

  public setPoints(points: number[][]) {
    this.points = points.map((v) => v.slice()) // make copy
  }

  public setPause(pause: boolean) {
    this.pause = pause
  }

  public setAutoSIR(autoSIR: boolean) {
    this.autoSIR = autoSIR
  }

  public setComponentEnable(componentEnable: Map<string, boolean>) {
    this.componentEnable = new Map(componentEnable.entries())
  }

  public getPosterior(): Map<string, XDistribution> {
    return new Map(
      Array.from(this.stats.entries()).map(([k, v]) => [k, v.summarize()]),
    )
  }

  public Reset() {
    this.totalFailedSamples = 0
    this.frameCount = 0
    this.t0 = performance.now()
  }

  // Sets up and runs the inference animation. Returns a function which can
  // be used to halt the animation (after the current frame is rendered).
  public run(): () => void {
    const maxSamplesPerParticle = 100_000
    // XXX: could get the above two constants by looking at the HTML,
    // but we really should learn to use a framework at some point
    const gpu = new GPGPU_Inference(
      this.modelParameters.size,
      maxSamplesPerParticle,
    )
    const renderer = new Render(this.modelParameters.size)

    let stopAnimation = false
    this.t0 = performance.now() // TODO: need to reset this from time to time along with frame count

    const frame = (t: DOMHighResTimeStamp) => {
      let result = undefined
      try {
        if (!this.pause) {
          result = gpu.inference(
            {
              points: this.points,
              coefficients: this.modelParameters,
              component_enable: this.componentEnable,
            },
            this.inferenceParameters,
          )
        }
        if (result) {
          this.totalFailedSamples += result.failedSamples

          const pOutlierStats = new RunningStats()

          for (const m of result.selectedModels) {
            let i = 0
            for (const v of this.stats.values()) {
              v.observe(m.model[i++])
            }
            pOutlierStats.observe(m.p_outlier)
          }
          ++this.frameCount
          const fps = Math.trunc(this.frameCount / ((t - this.t0) / 1e3))
          renderer.render(this.points, result)
          const info = {
            totalFailedSamples: this.totalFailedSamples,
            ips: result.ips,
            fps: fps,
            autoSIR: this.autoSIR,
            pOutlierStats: pOutlierStats,
          }
          this.inferenceReportCallback(info)
          // emptyPosterior.innerText = totalFailedSamples.toString()
        }
        if (stopAnimation) {
          console.log("halting animation")
          return
        }
        requestAnimationFrame(frame)
      } catch (error) {
        log("error", error)
      }
    }
    console.log("starting animation")
    requestAnimationFrame(frame)
    return () => {
      console.log("requesting animation halt")
      stopAnimation = true
    }
  }
}
