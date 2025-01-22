import {
  GPGPU_Inference,
  InferenceParameters,
  InferenceResult,
} from "./gpgpu.ts"
import { RunningStats, XDistribution } from "./stats.ts"
import { TypedObject } from "./utils"

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
  private totalFailedSamples = 0
  private gpu: GPGPU_Inference
  private result: InferenceResult
  vizInlierSigma: boolean = false
  private frameTimeBuf: number[] = []

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
    this.gpu = new GPGPU_Inference(
      Object.keys(this.modelParameters).length,
      this.maxSamplesPerParticle,
    )
    console.log("created an animator")
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
        m.drift_coefficient(
          i,
          0.02,
          p,
          this.modelParameters.inlier.get("mu"),
          this.points,
        ),
      )
    })
  }

  public Reset() {
    this.result.selectedModels = []
    this.totalFailedSamples = 0
    this.frameTimeBuf.length = 0
  }

  public awaitResult() {
    if (!this.pause) {
      if (this.autoDrift) {
        if (!this.result.selectedModels.length) {
          this.result = this.gpu.inference(
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
        this.result = this.gpu.inference(
          {
            points: this.points,
            coefficients: this.modelParameters,
            component_enable: this.componentEnable,
          },
          this.inferenceParameters,
        )
      }
    }
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
    const now = performance.now()
    this.frameTimeBuf.push(now)
    while (this.frameTimeBuf.length && this.frameTimeBuf[0] < now - 1000)
      this.frameTimeBuf.shift()
    const fps = this.frameTimeBuf.length
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
    return info
  }

  // // Sets up and runs the inference animation. Returns a function which can
  // // be used to halt the animation (after the current frame is done).
  // public run(): () => void {
  //   let stopAnimation = false
  //   this.t0 = performance.now() // TODO: need to reset this from time to time along with frame count

  //   const frame = (elapsedTime: number) => {

  //       if (stopAnimation) {
  //         console.log("halting animation")
  //         return
  //       }
  //       requestAnimationFrame(frame)
  //     } catch (error) {
  //       log("error", error)
  //     }
  //   }
  //   console.log("starting animation")
  //   requestAnimationFrame(frame)
  //   return () => {
  //     console.log("requesting animation halt")
  //     stopAnimation = true
  //   }
  // }
}
