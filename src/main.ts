import './style.css'
// import { setupCounter } from './counter.ts'
import { GPGPU_Inference } from './gpgpu.ts'
import { Render } from './render.ts'
import { Normal } from './model.ts'



const MODEL_SIZE = 3

const points = [-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4].map(x => [x, .7*x - .2 + 0.2 *x*x])
points[2][1] = +.9

function log(level: string, message: any): void {
  if (level == 'error') {
    console.error(message)
  } else {
    console.info(message)
  }
  const d = document.createElement('div')
  d.className = 'log-' + level
  const t = document.createTextNode(message.toString())
  d.appendChild(t)
  document.querySelector('#app')?.appendChild(d)
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

  observe(value: number) {
    this.count += 1
    const delta = value - this.mean
    this.mean += delta / this.count
    const delta2 = value - this.mean
    this.m2 += delta * delta2
  }

  summarize() {
    return {
      mu: this.mean,
      sigma: Math.sqrt(this.m2 / (this.count - 1))
    }
  }

  reset() {
    this.count = 0;
    this.mean = 0;
    this.m2 = 0;
  }
}

const initial_alpha = () => [
  {mu: 0, sigma: 2},
  {mu: 0, sigma: 2},
  {mu: 0, sigma: 2}
]

function main() {
  let samples_per_batch = 10000
  let batches_per_frame = 20
  const stats = Array.from({ length: MODEL_SIZE }, () => new RunningStats)

  let alpha: Normal[] = initial_alpha()

  function setupSlider(element_name: string, effect: (value: number) => void) {
    const elt = document.querySelector<HTMLInputElement>(element_name)
    const v_elt = document.querySelector<HTMLSpanElement>(element_name + '-value')
    if (elt) {
      elt.addEventListener('input', event => {
        const target = event.target as HTMLInputElement
        effect(target.valueAsNumber)
        if (v_elt) {
          v_elt.innerText = target.value
        }
      })
      if (v_elt) {
        v_elt.innerText = elt.valueAsNumber.toFixed(2)
      }
    }
    return elt
  }

  setupSlider('#a0_mu', v => { alpha[0].mu = v })
  setupSlider('#a0_sigma', v => { alpha[0].sigma = v })
  setupSlider('#a1_mu', v => { alpha[1].mu = v })
  setupSlider('#a1_sigma', v => { alpha[1].sigma = v })
  setupSlider('#a2_mu', v => { alpha[2].mu = v })
  setupSlider('#a2_sigma', v => { alpha[2].sigma = v })

  const gpu = new GPGPU_Inference(samples_per_batch)
  const renderer = new Render()

  let point_eviction_index = 0;
  renderer.canvas.addEventListener('click', event => {
    const target = event.target as HTMLCanvasElement
    const rect = target.getBoundingClientRect()
    const x = (event.clientX - rect.left) / target.width * 2.0 - 1.0
    const y = (event.clientY - rect.top) / target.height * -2.0 + 1.0

    points[point_eviction_index][0] = x
    points[point_eviction_index][1] = y
    if (++point_eviction_index >= points.length) {
      point_eviction_index = 0
    }
  })

  function setSliderValues(values: Normal[]) {
    for (let i = 0; i < MODEL_SIZE; ++i) {
      for (let k of ['mu', 'sigma']) {
        const elt = document.querySelector<HTMLInputElement>(`#a${i}_${k}`)!
        elt.value = values[i][k as keyof Normal].toFixed(2).toString()
        elt.dispatchEvent(new CustomEvent('input'))
      }
      stats[i].reset()
    }
  }

  document.querySelector<HTMLButtonElement>('#sir')?.addEventListener('click', () => {
    setSliderValues(stats.map(s => s.summarize(), [stats.keys()]))
  })

  document.querySelector<HTMLButtonElement>('#reset-priors')?.addEventListener('click', () => {
    alpha = initial_alpha()
    setSliderValues(alpha)
  })

  let frame_count = 0;
  let t0 = 0;

  function frame(t: DOMHighResTimeStamp) {
    try {
      const { selected_models, ips } = gpu.inference(batches_per_frame, {
        points: points,
        alpha: alpha
      })

      for (const m of selected_models) {
        for (let i = 0; i < MODEL_SIZE; ++i) {
          stats[i].observe(m.model[i])
        }
      }
      if (t0 == 0) {
        t0 = t
      }
      ++frame_count
      if (frame_count % 200 == 0) {
        const fps = Math.trunc(frame_count / ((t - t0)/1e3))
        document.querySelector<HTMLSpanElement>('#fps')!.innerText = fps.toString()
        document.querySelector<HTMLSpanElement>('#ips')!.innerText = `${(ips / 1e6).toFixed(1)} M`
        frame_count = 0
        t0 = 0
      }
      if (frame_count % 50 == 0) {
        for (let i = 0; i < MODEL_SIZE; ++i) {
          const s = stats[i].summarize()
          document.querySelector<HTMLSpanElement>(`#a${i}_mu-posterior`)!.innerText = s.mu.toFixed(2).toString()
          document.querySelector<HTMLSpanElement>(`#a${i}_sigma-posterior`)!.innerText = s.sigma.toFixed(2).toString()
        }
      }
      // const ols = selected_models.map(m => m.p_outlier.toFixed(2).toString()).join(', ')
      // document.querySelector<HTMLSpanElement>('#p_outlier')!.innerText = ols
      renderer.render(points, selected_models)
      requestAnimationFrame(frame)
    } catch (error) {
      log('error', error)
    }
  }

  requestAnimationFrame(frame)
}

try {
  main()
} catch (error) {
  log('error', error)
}
