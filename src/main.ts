// import { setupCounter } from './counter.ts'
import { GPGPU_Inference } from "./gpgpu.ts"
import { Render } from "./render.ts"
import type { Normal } from "./model.ts"

const xs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
const points = xs.map((x) => [x, 0.7 * x + 0.3 * Math.sin(9.0 * x + 0.3)])
points[7][1] = 0.75

// (x) => [x, 0.7 * x - 0.2 + 0.2 * x * x],
// points[2][1] = +0.9

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

const params = [
  { name: "a_0", initialValue: { mu: 0, sigma: 2 } },
  { name: "a_1", initialValue: { mu: 0, sigma: 2 } },
  { name: "a_2", initialValue: { mu: 0, sigma: 2 } },
  { name: "omega", initialValue: { mu: 0, sigma: 2 } },
  { name: "A", initialValue: { mu: 0, sigma: 2 } },
  { name: "phi", initialValue: { mu: 0, sigma: 2 } },
  { name: "inlier", initialValue: { mu: 0.3, sigma: 0.07 } }, // TODO: change to uniform
]


const MODEL_SIZE = params.length

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

const initialAlpha = () => params.map((p) => Object.assign({}, p.initialValue))

// Sets up and runs the inference animation. Returns a function which can
// be used to halt the animation (after the current frame is rendered).
export function run(setInfo, uiParams): () => void {
  console.log('HERE')
  const inferenceParameters = uiParams().inferenceParameters
  const stats = params.map(() => new RunningStats())

  let alpha: Normal[] = initialAlpha()

  const maxSamplesPerParticle = 100_000
  // XXX: could get the above two constants by looking at the HTML,
  // but we really should learn to use a framework at some point
  const gpu = new GPGPU_Inference(params.length, maxSamplesPerParticle)
  const renderer = new Render(params.length)

  let pointEvictionIndex = 0
  renderer.canvas.addEventListener("click", (event) => {
    const target = event.target as HTMLCanvasElement
    const rect = target.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / target.width) * 2.0 - 1.0
    const y = ((event.clientY - rect.top) / target.height) * -2.0 + 1.0

    points[pointEvictionIndex][0] = x
    points[pointEvictionIndex][1] = y
    if (++pointEvictionIndex >= points.length) {
      pointEvictionIndex = 0
    }
    Reset()
  })

  function setSliderValues(values: Normal[]): void {
    for (let i = 0; i < MODEL_SIZE; ++i) {
      for (const k of ["mu", "sigma"]) {
        const elt = document.querySelector<HTMLInputElement>(
          `#${params[i].name}_${k}`,
        )
        if (elt != null) {
          elt.value = values[i][k as keyof Normal].toFixed(2).toString()
          elt.dispatchEvent(new CustomEvent("input"))
        }
      }
      stats[i].reset()
    }
  }

  let stopAnimation = false

  function stop() {
    console.log('requesting animation halt')
    stopAnimation = true;
  }

  function SIR_Update() {
    setSliderValues(stats.map((s) => s.summarize(), [stats.keys()]))
  }

  document
    .querySelector<HTMLButtonElement>("#sir")
    ?.addEventListener("click", SIR_Update)

  function Reset() {
    alpha = initialAlpha()
    setSliderValues(alpha)
    totalFailedSamples = 0
    pause.checked = false
  }

  document
    .querySelector<HTMLButtonElement>("#reset-priors")
    ?.addEventListener("click", Reset)

  const pause = document.querySelector<HTMLInputElement>("#pause")!
  const autoSIR = document.querySelector<HTMLInputElement>("#auto-SIR")!

  let frameCount = 0
  let t0 = performance.now()
  let totalFailedSamples = 0

  function frame(t: DOMHighResTimeStamp): void {
    let result = undefined
    try {
      if (!pause.checked) {
        console.log('alpha', alpha)
        result = gpu.inference(
          {
            points,
            coefficients: alpha,
            component_enable: new Map().set('polynomial', true).set('foo', true),
          },
          inferenceParameters,
        )
      }
      if (result) {
        totalFailedSamples += result.failedSamples

        for (const m of result.selectedModels) {
          for (let j = 0; j < MODEL_SIZE; ++j) {
            stats[j].observe(m.model[j])
          }
        }
        ++frameCount
        const fps = Math.trunc(frameCount / ((t - t0) / 1e3))
        console.log('fps',fps)
        renderer.render(points, result)
        const info = {
          totalFailedSamples: totalFailedSamples,
          ips: result.ips,
          fps: fps,
          posterior: stats.map((s, i) => [params[i].name, s.summarize()]),
        }
        setInfo(info)
        // emptyPosterior.innerText = totalFailedSamples.toString()
        if (autoSIR.checked) {
          SIR_Update()
        }
      }
      if (stopAnimation) {
        console.log('halting animation')
        return
      }
      if (frameCount < 1000) requestAnimationFrame(frame)
    } catch (error) {
      log("error", error)
    }
  }
  console.log('starting animation')
  requestAnimationFrame(frame)
  return stop
}

// try {
//   main()
// } catch (error) {
//   log("error", error)
// }
