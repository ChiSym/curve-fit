import "./style.css"
// import { setupCounter } from './counter.ts'
import { GPGPU_Inference } from "./gpgpu.ts"
import { Render } from "./render.ts"
import type { Normal } from "./model.ts"

const MODEL_SIZE = 3

const points = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4].map(
  (x) => [x, 0.7 * x - 0.2 + 0.2 * x * x],
)
points[2][1] = +0.9

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

const initialAlpha = (): NormalParams[] => [
  { mu: 0, sigma: 2 },
  { mu: 0, sigma: 2 },
  { mu: 0, sigma: 2 },
]

function main(): void {
  const samplesPerBatch = 10000
  const batchesPerFrame = 20
  const stats = Array.from({ length: MODEL_SIZE }, () => new RunningStats())

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
    }
  }

  setupSlider("#a0_mu", (v) => {
    alpha[0].mu = v
  })
  setupSlider("#a0_sigma", (v) => {
    alpha[0].sigma = v
  })
  setupSlider("#a1_mu", (v) => {
    alpha[1].mu = v
  })
  setupSlider("#a1_sigma", (v) => {
    alpha[1].sigma = v
  })
  setupSlider("#a2_mu", (v) => {
    alpha[2].mu = v
  })
  setupSlider("#a2_sigma", (v) => {
    alpha[2].sigma = v
  })

  const gpu = new GPGPU_Inference(samplesPerBatch)
  const renderer = new Render()

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
  })

  function setSliderValues(values: Normal[]): void {
    for (let i = 0; i < MODEL_SIZE; ++i) {
      for (const k of ["mu", "sigma"]) {
        const elt = document.querySelector<HTMLInputElement>(`#a${i}_${k}`)
        if (elt != null) {
          elt.value = values[i][k as keyof Normal].toFixed(2).toString()
          elt.dispatchEvent(new CustomEvent("input"))
        }
      }
      stats[i].reset()
    }
  }

  document
    .querySelector<HTMLButtonElement>("#sir")
    ?.addEventListener("click", () => {
      setSliderValues(stats.map((s) => s.summarize(), [stats.keys()]))
    })

  document
    .querySelector<HTMLButtonElement>("#reset-priors")
    ?.addEventListener("click", () => {
      alpha = initialAlpha()
      setSliderValues(alpha)
    })

  let frameCount = 0
  let t0 = 0

  function frame(t: DOMHighResTimeStamp): void {
    try {
      const { selectedModels, ips } = gpu.inference(batchesPerFrame, {
        points,
        alpha,
      })

      for (const m of selectedModels) {
        for (let i = 0; i < MODEL_SIZE; ++i) {
          stats[i].observe(m.model[i])
        }
      }
      if (t0 === 0) {
        t0 = t
      }
      ++frameCount
      if (frameCount % 200 === 0) {
        const fps = Math.trunc(frameCount / ((t - t0) / 1e3))
        setInnerText("#fps", fps.toString())
        setInnerText("#ips", `${(ips / 1e6).toFixed(1)} M`)
        frameCount = 0
        t0 = 0
      }
      if (frameCount % 50 === 0) {
        for (let i = 0; i < MODEL_SIZE; ++i) {
          const s = stats[i].summarize()
          setInnerText(`#a${i}_mu-posterior`, s.mu.toFixed(2).toString())
          setInnerText(`#a${i}_sigma-posterior`, s.sigma.toFixed(2).toString())
        }
      }
      // const ols = selected_models.map(m => m.p_outlier.toFixed(2).toString()).join(', ')
      // document.querySelector<HTMLSpanElement>('#p_outlier')!.innerText = ols
      renderer.render(points, selectedModels)
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
