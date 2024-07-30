import "./style.css"
// import { setupCounter } from './counter.ts'
import { GPGPU_Inference, InferenceParameters } from "./gpgpu.ts"
import { Render } from "./render.ts"
import katex from "katex"
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

interface NormalParams {
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

const model_components = ["polynomial", "periodic"]

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

function main(): void {
  const stats = params.map(() => new RunningStats())
  const modelEnable = new Map()
  const inferenceParameters: InferenceParameters = {
    numParticles: 0,
    importanceSamplesPerParticle: 0,
  }

  let alpha: Normal[] = initialAlpha()

  function setInnerText(selector: string, text: string): void {
    const elt = document.querySelector<HTMLSpanElement>(selector)
    if (elt == null) throw new Error(`unable to find ${selector}`)
    elt.innerText = text
  }

  function setupSlider(
    elementName: string,
    effect: (value: number) => void,
  ): void {
    const elt = document.querySelector<HTMLInputElement>(elementName)
    const vElt = document.querySelector<HTMLSpanElement>(elementName + "-value")
    if (elt != null) {
      elt.addEventListener("input", (event) => {
        const target = event.target as HTMLInputElement
        effect(target.valueAsNumber)
        if (vElt != null) {
          vElt.innerText = target.value
        }
      })
      if (vElt != null) {
        vElt.innerText = elt.valueAsNumber.toFixed(2)
      }
    } else {
      console.log(`cannot find ${elementName}`)
    }
  }

  model_components.forEach((m) => {
    const elt = document.querySelector<HTMLInputElement>("#" + m + "_enable")
    if (elt) {
      modelEnable.set(m, true)
      elt.checked = true
      elt.addEventListener("change", () => {
        modelEnable.set(m, elt.checked)
        console.log(modelEnable)
      })
    } else console.log(`can't find ${m}`)
  })

  Object.keys(inferenceParameters).forEach((m) => {
    const elt = document.querySelector<HTMLSelectElement>("#" + m)
    if (elt) {
      elt.addEventListener("change", () => {
        console.log(`${m} -> ${parseInt(elt.value)}`)
        inferenceParameters[m as keyof InferenceParameters] = parseInt(
          elt.value,
        )
      })
      elt.dispatchEvent(new CustomEvent("change"))
    }
  })

  params.forEach((p, i) => {
    setupSlider("#" + p.name + "_mu", (v) => {
      alpha[i].mu = v
    })
    setupSlider("#" + p.name + "_sigma", (v) => {
      alpha[i].sigma = v
    })
  })

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

  const emptyPosterior =
    document.querySelector<HTMLSpanElement>("#empty-posterior")!
  const pause = document.querySelector<HTMLInputElement>("#pause")!
  const autoSIR = document.querySelector<HTMLInputElement>("#auto-SIR")!

  // render math
  const mathElements = document.querySelectorAll<HTMLElement>(".katex")
  Array.from(mathElements).forEach((el) => {
    if (el.textContent) {
      katex.render(el.textContent, el, {
        throwOnError: false,
      })
    }
  })
  let frameCount = 0
  let t0 = 0
  let totalFailedSamples = 0

  function frame(t: DOMHighResTimeStamp): void {
    let result = undefined
    try {
      if (!pause.checked) {
        result = gpu.inference(
          {
            points,
            coefficients: alpha,
            component_enable: modelEnable,
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
        if (t0 === 0) {
          t0 = t
        }
        ++frameCount
        if (frameCount % 200 === 0) {
          const fps = Math.trunc(frameCount / ((t - t0) / 1e3))
          setInnerText("#fps", fps.toString())
          setInnerText("#ips", `${(result.ips / 1e6).toFixed(1)} M`)
          frameCount = 0
          t0 = 0
        }
        if (frameCount % 50 === 0) {
          for (let i = 0; i < MODEL_SIZE; ++i) {
            const s = stats[i].summarize()
            setInnerText(
              `#${params[i].name}_mu-posterior`,
              s.mu.toFixed(2).toString(),
            )
            setInnerText(
              `#${params[i].name}_sigma-posterior`,
              s.sigma.toFixed(2).toString(),
            )
          }
        }
        // const ols = selected_models.map(m => m.p_outlier.toFixed(2).toString()).join(', ')
        // document.querySelector<HTMLSpanElement>('#p_outlier')!.innerText = ols
        renderer.render(points, result)
        emptyPosterior.innerText = totalFailedSamples.toString()
        if (autoSIR.checked) {
          SIR_Update()
        }
      }
      requestAnimationFrame(frame)
    } catch (error) {
      log("error", error)
    }
  }
  requestAnimationFrame(frame)
}

try {
  main()
} catch (error) {
  log("error", error)
}
