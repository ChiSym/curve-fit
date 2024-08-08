import { GPGPU_Inference, InferenceParameters } from "./gpgpu.ts"
import { Render } from "./render.ts"
import { RunningStats } from "./stats.ts"

export interface Distribution {
  mu: number,
  sigma: number
}

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
  posterior: Map<string, RunningStats>
  autoSIR: boolean
}

export class Animator {

  private modelParameters: Map<string, Distribution>
  private readonly inferenceParameters: InferenceParameters
  private readonly inferenceReportCallback: (r: InferenceReport) => void
  private readonly stats: Map<string, RunningStats>
  private points: number[][]
  private pause: boolean = false
  private autoSIR: boolean = false


  // TODO: define type ModelParameters as Map<string, Distribution>; change signature of inference
  // engine code to take multiple parameters, giving up old name

  constructor(modelParameters: Map<string, Distribution>, inferenceParameters: InferenceParameters, inferenceReportCallback: (r: InferenceReport) => void) {
    this.inferenceReportCallback = inferenceReportCallback
    // make copies of the initial values
    this.modelParameters = new Map(modelParameters.entries())
    this.inferenceParameters = Object.assign({}, inferenceParameters)
    this.stats = new Map(Array.from(modelParameters.keys()).map(k => [k, new RunningStats]))
  }

  public setInferenceParameters(ps: InferenceParameters) {
    this.inferenceParameters.importanceSamplesPerParticle = ps.importanceSamplesPerParticle
    this.inferenceParameters.numParticles = ps.numParticles
  }

  public setModelParameters(params: Map<string, Distribution>) {
    this.modelParameters = new Map(Array.from(params.entries()).map(([k, v]) => [k, Object.assign({}, v)]))
    this.stats.forEach(s => s.reset())
  }

  public setPoints(points: number[][]) {
    this.points = points.map(v => v.slice())  // make copy
  }

  public setPause(pause: boolean) {
    this.pause = pause
  }

  public setAutoSIR(autoSIR: boolean) {
    this.autoSIR = autoSIR
  }

  // Sets up and runs the inference animation. Returns a function which can
  // be used to halt the animation (after the current frame is rendered).
  public run(): () => void {
    const maxSamplesPerParticle = 100_000
    // XXX: could get the above two constants by looking at the HTML,
    // but we really should learn to use a framework at some point
    const gpu = new GPGPU_Inference(this.modelParameters.size, maxSamplesPerParticle)
    const renderer = new Render(this.modelParameters.size)

    let stopAnimation = false

    function Reset() {
      totalFailedSamples = 0
    }

    let frameCount = 0
    let t0 = performance.now() // TODO: need to reset this from time to time along with frame count
    let totalFailedSamples = 0

    const frame = (t: DOMHighResTimeStamp) => {
      let result = undefined
      try {
        if (!this.pause) {
          result = gpu.inference(
            {
              points: this.points,
              coefficients: this.modelParameters,
              component_enable: new Map().set('polynomial', true).set('foo', true),
            },
            this.inferenceParameters,
          )
        }
        if (result) {
          totalFailedSamples += result.failedSamples

          for (const m of result.selectedModels) {
            let i = 0;
            for (const v of this.stats.values()) {
              v.observe(m.model[i++])
            }
          }
          ++frameCount
          const fps = Math.trunc(frameCount / ((t - t0) / 1e3))
          renderer.render(this.points, result)
          const info = {
            totalFailedSamples: totalFailedSamples,
            ips: result.ips,
            fps: fps,
            posterior: this.stats,
            autoSIR: this.autoSIR
          }
          this.inferenceReportCallback(info)
          // emptyPosterior.innerText = totalFailedSamples.toString()
        }
        if (stopAnimation) {
          console.log('halting animation')
          return
        }
        requestAnimationFrame(frame)
      } catch (error) {
        log("error", error)
      }
    }
    console.log('starting animation')
    requestAnimationFrame(frame)
    return () => {
      console.log('requesting animation halt')
      stopAnimation = true;
    }
  }
}
