import {
  GPGPU_Inference,
  InferenceParameters,
  InferenceResult,
} from "./gpgpu.ts"
import { RunningStats, XDistribution } from "./stats.ts"
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
  document.querySelector("#root")?.appendChild(d)
}

export interface InferenceReport {
  fps: number
  totalFailedSamples: number
  pOutlierStats: RunningStats
  inlierSigmaStats: RunningStats
  autoSIR: boolean
  autoDrift: boolean
  inferenceResult: InferenceResult
}

export class Animator {
  private modelParameters: TypedObject<XDistribution>
  private readonly inferenceParameters: InferenceParameters
  private readonly inferenceReportCallback: (r: InferenceReport) => void
  private readonly stats: TypedObject<RunningStats>
  private readonly maxSamplesPerParticle: number
  private points: number[][] = []
  private pause: boolean = false
  private autoSIR: boolean = false
  private autoDrift: boolean = false
  private componentEnable: TypedObject<boolean> = {}
  private frameCount = 0
  private totalFailedSamples = 0
  private t0: DOMHighResTimeStamp = performance.now()
  private result: InferenceResult
  vizInlierSigma: boolean = false

  constructor(
    modelParameters: TypedObject<XDistribution>,
    inferenceParameters: InferenceParameters,
    maxSamplesPerParticle: number,
    inferenceReportCallback: (r: InferenceReport) => void,
  ) {
    this.inferenceReportCallback = inferenceReportCallback
    // make copies of the initial values
    this.modelParameters = { ...modelParameters }
    this.inferenceParameters = { ...inferenceParameters }
    this.maxSamplesPerParticle = maxSamplesPerParticle
    this.stats = Object.keys(modelParameters).reduce((acc, k) => {
      acc[k] = new RunningStats()
      return acc
    }, {} as TypedObject<RunningStats>)
    this.result = {
      selectedModels: [],
      ips: 0,
      failedSamples: 0,
    }
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

  public setAutoDrift(autoDrift: boolean) {
    this.autoDrift = autoDrift
  }

  public setComponentEnable(componentEnable: TypedObject<boolean>) {
    this.componentEnable = { ...componentEnable }
  }

  public getPosterior(): TypedObject<XDistribution> {
    return Object.fromEntries(
      Object.entries(this.stats).map(([k, v]) => [k, v.summarize()]),
    )
  }

  public Drift() {
    this.result.selectedModels.forEach((m) => {
      //console.log(this.modelParameters)
      // m.drift_coefficient(
      //   0,
      //   0.005,
      //   this.modelParameters.a_0,
      //   this.modelParameters.inlier.get("mu"),
      //   this.points,
      // )
      Object.values(this.modelParameters).forEach((p, i) =>
        m.drift_coefficient(i, 0.02, p, this.modelParameters.inlier.get('mu'), this.points)
      )
    })
  }

  public Reset() {
    this.result.selectedModels = []
    this.totalFailedSamples = 0
    this.frameCount = 0
    this.t0 = performance.now()
  }

  // Sets up and runs the inference animation. Returns a function which can
  // be used to halt the animation (after the current frame is done).
  public run(): () => void {
    // XXX: could get the above two constants by looking at the HTML,
    // but we really should learn to use a framework at some point
    const gpu = new GPGPU_Inference(
      Object.keys(this.modelParameters).length,
      this.maxSamplesPerParticle,
    )

    let stopAnimation = false
    this.t0 = performance.now() // TODO: need to reset this from time to time along with frame count

    const frame = (t: DOMHighResTimeStamp) => {
      try {
        if (!this.pause) {
          if (this.autoDrift) {
            if (!this.result.selectedModels.length) {
              this.result = gpu.inference(
                {
                  points: this.points,
                  coefficients: this.modelParameters,
                  component_enable: this.componentEnable,
                },
                this.inferenceParameters,
              )
            }
            this.Drift()
          } else {
            this.result = gpu.inference(
              {
                points: this.points,
                coefficients: this.modelParameters,
                component_enable: this.componentEnable,
              },
              this.inferenceParameters,
            )
          }
        }
        if (this.result) {
          this.totalFailedSamples += this.result.failedSamples

          const pOutlierStats = new RunningStats()
          const inlierSigmaStats = new RunningStats()

          for (const m of this.result.selectedModels) {
            let i = 0
            for (const v of Object.values(this.stats)) {
              v.observe(m.model[i++])
            }
            pOutlierStats.observe(m.p_outlier)
            inlierSigmaStats.observe(m.inlier_sigma)
          }
          ++this.frameCount
          const fps = Math.trunc(this.frameCount / ((t - this.t0) / 1e3))
          const info = {
            totalFailedSamples: this.totalFailedSamples,
            fps: fps,
            autoSIR: this.autoSIR,
            autoDrift: this.autoDrift,
            pOutlierStats: pOutlierStats,
            inlierSigmaStats: inlierSigmaStats,
            inferenceResult: this.result, // TODO: make a getter
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
