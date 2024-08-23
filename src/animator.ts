import { XDistribution } from "./App.tsx"
import { GPGPU_Inference, InferenceParameters } from "./gpgpu.ts"
import { Render } from "./render.ts"
import { RunningStats } from "./stats.ts"
import { TypedObject } from "./utils"

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
  private modelParameters: TypedObject<XDistribution>
  private readonly inferenceParameters: InferenceParameters
  private readonly inferenceReportCallback: (r: InferenceReport) => void
  private readonly stats: TypedObject<RunningStats>
  private points: number[][] = []
  private pause: boolean = false
  private autoSIR: boolean = false
  private visualizeInlierSigma: boolean = false
  private componentEnable: TypedObject<boolean> = {}
  private frameCount = 0
  private totalFailedSamples = 0
  private t0: DOMHighResTimeStamp = performance.now()
  private canvas: HTMLCanvasElement
  // TODO: define type ModelParameters as Map<string, Distribution>; change signature of inference
  // engine code to take multiple parameters, giving up old name

  constructor(
    canvas: HTMLCanvasElement,
    modelParameters: TypedObject<XDistribution>,
    inferenceParameters: InferenceParameters,
    inferenceReportCallback: (r: InferenceReport) => void,
  ) {
    this.canvas = canvas
    this.inferenceReportCallback = inferenceReportCallback
    // make copies of the initial values
    this.modelParameters = { ...modelParameters }
    this.inferenceParameters = { ...inferenceParameters }
    this.stats = Object.keys(modelParameters).reduce((acc, k) => {
      acc[k] = new RunningStats()
      return acc
    }, {} as TypedObject<RunningStats>)
  }

  public setInferenceParameters(ps: InferenceParameters) {
    this.inferenceParameters.importanceSamplesPerParticle =
      ps.importanceSamplesPerParticle
    this.inferenceParameters.numParticles = ps.numParticles
    this.Reset()
  }

  public setModelParameters(params: TypedObject<XDistribution>) {
    this.modelParameters = { ...params }
    Object.values(this.stats).forEach((s) => s.reset())
  }

  public setPoints(points: number[][]) {
    this.points = points.map((v) => [...v]) // make copy
  }

  public setPause(pause: boolean) {
    this.pause = pause
  }

  public setAutoSIR(autoSIR: boolean) {
    this.autoSIR = autoSIR
  }

  public setVisualizeInlierSigma(vis: boolean) {
    this.visualizeInlierSigma = vis
  }

  public setComponentEnable(componentEnable: TypedObject<boolean>) {
    this.componentEnable = { ...componentEnable }
  }

  public getPosterior(): TypedObject<XDistribution> {
    return Object.fromEntries(
      Object.entries(this.stats).map(([k, v]) => [k, v.summarize()]),
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
      Object.keys(this.modelParameters).length,
      maxSamplesPerParticle,
    )
    const renderer = new Render(
      this.canvas,
      Object.keys(this.modelParameters).length,
    )

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
            for (const v of Object.values(this.stats)) {
              v.observe(m.model[i++])
            }
            pOutlierStats.observe(m.p_outlier)
          }
          ++this.frameCount
          const fps = Math.trunc(this.frameCount / ((t - this.t0) / 1e3))
          renderer.render(
            {
              points: this.points,
              visualizeInlierSigma: this.visualizeInlierSigma,
            },
            result,
          )
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
