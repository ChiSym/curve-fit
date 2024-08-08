import { GPGPU_Inference, InferenceParameters } from "./gpgpu.ts"
import { Render } from "./render.ts"

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

export interface NormalParams {
  mu: number
  sigma: number
}

export interface InferenceReport {
  fps: number
  ips: number
  totalFailedSamples: number
  posterior: Map<string, RunningStats>
}

class RunningStats {
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

export class Animator {

  private modelParameters: Map<string, Distribution>
  private readonly inferenceParameters: InferenceParameters
  private readonly points: number[][]
  private readonly inferenceReportCallback: (r: InferenceReport) => void
  private readonly stats: Map<string, RunningStats>


  // TODO: define type ModelParameters as Map<string, Distribution>; change signature of inference
  // engine code to take multiple parameters, giving up old name

  constructor(modelParameters: Map<string, Distribution>, inferenceParameters: InferenceParameters, inferenceReportCallback: (r: InferenceReport) => void) {
    const xs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
    this.points = xs.map((x) => [x, 0.7 * x + 0.3 * Math.sin(9.0 * x + 0.3)])
    this.points[7][1] = 0.75
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
    this.modelParameters = new Map(params.entries())
  }

  public setPoints() {

  }

  // Sets up and runs the inference animation. Returns a function which can
  // be used to halt the animation (after the current frame is rendered).
  public run(): () => void {
    console.log('HERE')
    const maxSamplesPerParticle = 100_000
    // XXX: could get the above two constants by looking at the HTML,
    // but we really should learn to use a framework at some point
    const gpu = new GPGPU_Inference(this.modelParameters.size, maxSamplesPerParticle)
    const renderer = new Render(this.modelParameters.size)

    let pointEvictionIndex = 0
    renderer.canvas.addEventListener("click", (event) => {
      const target = event.target as HTMLCanvasElement
      const rect = target.getBoundingClientRect()
      const x = ((event.clientX - rect.left) / target.width) * 2.0 - 1.0
      const y = ((event.clientY - rect.top) / target.height) * -2.0 + 1.0

      this.points[pointEvictionIndex][0] = x
      this.points[pointEvictionIndex][1] = y
      if (++pointEvictionIndex >= this.points.length) {
        pointEvictionIndex = 0
      }
      Reset()
    })

    // function setSliderValues(values: Normal[]): void {
    //   for (let i = 0; i < MODEL_SIZE; ++i) {
    //     for (const k of ["mu", "sigma"]) {
    //       const elt = document.querySelector<HTMLInputElement>(
    //         `#${params[i].name}_${k}`,
    //       )
    //       if (elt != null) {
    //         elt.value = values[i][k as keyof Normal].toFixed(2).toString()
    //         elt.dispatchEvent(new CustomEvent("input"))
    //       }
    //     }
    //     stats[i].reset()
    //   }
    // }

    let stopAnimation = false

    function SIR_Update() {
      //setSliderValues(stats.map((s) => s.summarize(), [stats.keys()]))
    }

    document
      .querySelector<HTMLButtonElement>("#sir")
      ?.addEventListener("click", SIR_Update)

    function Reset() {
      totalFailedSamples = 0
      pause.checked = false
    }

    document
      .querySelector<HTMLButtonElement>("#reset-priors")
      ?.addEventListener("click", Reset)

    const pause = document.querySelector<HTMLInputElement>("#pause")!
    const autoSIR = document.querySelector<HTMLInputElement>("#auto-SIR")!

    let frameCount = 0
    let t0 = performance.now() // TODO: need to reset this from time to time along with frame count
    let totalFailedSamples = 0

    const frame = (t: DOMHighResTimeStamp) => {
      let result = undefined
      try {
        if (!pause.checked) {
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
          console.log('fps',fps)
          renderer.render(this.points, result)
          const info = {
            totalFailedSamples: totalFailedSamples,
            ips: result.ips,
            fps: fps,
            posterior: this.stats
          }
          this.inferenceReportCallback(info)
          // emptyPosterior.innerText = totalFailedSamples.toString()
          if (autoSIR.checked) {
            SIR_Update()
          }
        }
        if (stopAnimation) {
          console.log('halting animation')
          return
        }
        //requestAnimationFrame(frame)
      } catch (error) {
        log("error", error)
      }
    }
    console.log('starting animation')
    requestAnimationFrame(frame)
    return function() {
      console.log('requesting animation halt')
      stopAnimation = true;
    }
  }





}

// try {
//   main()
// } catch (error) {
//   log("error", error)
// }
