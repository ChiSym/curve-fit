import "./App.css"
import { Animator, Distribution } from "./animator.ts"
import { useState, useEffect, useRef, ChangeEvent } from "react"
import throttle from "lodash.throttle"
import katex from "katex"
import { InferenceParameters } from "./gpgpu.ts"
import { RunningStats } from "./stats.ts"

export const modelParams: Map<string, Distribution> = new Map([
  ["a_0", { mu: 0, sigma: 2 }],
  ["a_1", { mu: 0, sigma: 2 }],
  ["a_2", { mu: 0, sigma: 2 }],
  ["omega", { mu: 0, sigma: 2 }],
  ["A", { mu: 0, sigma: 2 }],
  ["phi", { mu: 0, sigma: 2 }],
  ["inlier", { mu: 0.3, sigma: 0.07 }], // TODO: change to uniform
])

const defaultInferenceParameters: InferenceParameters = {
  importanceSamplesPerParticle: 1000,
  numParticles: 10,
}

export default function CurveFit() {
  const animatorRef = useRef<Animator>(
    new Animator(modelParams, defaultInferenceParameters, setter),
  )

  const [inferenceParameters, setInferenceParameters] = useState(
    defaultInferenceParameters,
  )

  const [points, setPoints] = useState(() => {
    const xs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
    const points = xs.map((x) => [x, 0.7 * x + 0.3 * Math.sin(9.0 * x + 0.3)])
    points[7][1] = 0.75
    return { points: points, evictionIndex: 0 }
  })

  //const [modelParameters, setModelParameters] = useState([params.map(p => p.initialValue.mu), params.map(p=>p.initialValue.sigma))
  //  TODO: do we need an initial copy?
  const [emptyPosterior, setEmptyPosterior] = useState(0)
  const [modelState, setModelState] = useState(new Map(modelParams.entries()))
  const [posteriorState, setPosteriorState] = useState(
    new Map(modelParams.entries()),
  )

  const [componentEnable, setComponentEnable] = useState(
    new Map().set('polynomial', true).set('periodic', true)
  )

  const [ips, setIps] = useState(0.0)
  const [fps, setFps] = useState(0.0)

  function modelChange(k1: string, k2: string, v: number) {
    console.log(`${k1}_${k2} -> ${v}`)
    const old_value = modelState.get(k1)
    const updated_value = Object.assign({}, old_value)
    updated_value[k2 as keyof typeof updated_value] = v
    setModelState(new Map(modelState.entries()).set(k1, updated_value))
    animatorRef.current.setModelParameters(modelState)
  }

  const throttledSetIps = throttle(setIps, 500)
  const throttledSetFps = throttle(setFps, 500)
  const throttledSetPosteriorState = throttle(setPosteriorState, 500)

  function SIR_Update() {}

  function setter(data) {
    // This function is handed to the inference loop, which uses it to convey summary data
    // back to the UI.
    setEmptyPosterior(data.totalFailedSamples)
    throttledSetIps(data.ips)
    throttledSetFps(data.fps)
    const newState = (s: Map<string, RunningStats>) => {
      return new Map(
        Array.from(s.entries()).map(([k, v]) => [k, v.summarize()]),
      )
    }
    if (data.autoSIR) {
      const s = newState(data.posterior)
      setModelState(s)
      setPosteriorState(s)
      animatorRef.current.setModelParameters(s)
    } else {
      throttledSetPosteriorState(newState(data.posterior))
    }
  }

  useEffect(() => {
    // Things to do once the UI is set up:
    // - Render all the spans tagged with TeX source with KaTeX
    Array.from(document.querySelectorAll("span.katex-render")).forEach((e) => {
      const text = e.getAttribute("katex-source")!
      katex.render(text, e as HTMLElement)
    })
    // - Start the animation loop
    const a = animatorRef.current
    a.setInferenceParameters(inferenceParameters)
    a.setModelParameters(modelParams)
    a.setPoints(points.points)
    a.setComponentEnable(componentEnable)
    return a.run()
  }, [])

  function Reset() {
    setModelState(new Map(modelParams.entries()))
    setPosteriorState(new Map(modelParams.entries()))
    animatorRef.current.setModelParameters(modelParams)
    animatorRef.current.Reset()
  }

  function canvasClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const target = event.target as HTMLCanvasElement
    const rect = target.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / target.width) * 2.0 - 1.0
    const y = ((event.clientY - rect.top) / target.height) * -2.0 + 1.0

    const ps = points.points
    let i = points.evictionIndex
    ps[i][0] = x
    ps[i][1] = y
    if (++i >= ps.length) {
      i = 0
    }
    setPoints({ points: ps.slice(), evictionIndex: i })
    animatorRef.current.setPoints(ps)
    Reset()
  }

  return (
    <>
      <canvas id="c" onClick={canvasClick}></canvas>
      <br />
      FPS: <span id="fps">{fps}</span>
      <br />
      IPS: <span id="ips">{(ips / 1e6).toFixed(2) + " M"}</span>
      <br />
      <InferenceUI
        K={inferenceParameters.numParticles}
        N={inferenceParameters.importanceSamplesPerParticle}
        setK={(K: string) => {
          const newIP = { ...inferenceParameters, numParticles: parseInt(K) }
          setInferenceParameters(newIP)
          animatorRef.current.setInferenceParameters(newIP)
        }}
        setN={(N: string) => {
          const newIP = {
            ...inferenceParameters,
            importanceSamplesPerParticle: parseInt(N),
          }
          setInferenceParameters(newIP)
          animatorRef.current.setInferenceParameters(newIP)
        }}
      ></InferenceUI>
      <div id="model-components">
        <div className="column">
          <ModelComponent
            name="polynomial"
            enabled={componentEnable.get('polynomial')}
            onChange={e => {
              setComponentEnable(new Map(componentEnable.entries()).set('polynomial', e.target.checked))
              // XXX
            }}
            equation="a_0 + a_1 x + a_2 x^2"
          >
            {["a_0", "a_1", "a_2"].map((n) => (
              <ComponentParameter
                name={n}
                tex_name={n}
                value={modelState.get(n)}
                posterior_value={posteriorState.get(n)}
                onChange={modelChange}
              ></ComponentParameter>
            ))}
          </ModelComponent>
          <ModelComponent
            name="inlier (not working)"
            enabled={false}
            onChange={() => null}
            equation=""
          >
            <ComponentParameter
              name="inlier"
              tex_name="\sigma_\mathrm{in}"
              value={modelState.get("inlier")}
              posterior_value={posteriorState.get("inlier")}
              onChange={modelChange}
            ></ComponentParameter>
          </ModelComponent>
        </div>
        <div className="column">
          <ModelComponent
            name="periodic"
            enabled={componentEnable.get('periodic')}
            onChange={e => {
              const ce = new Map(componentEnable.entries()).set('periodic', e.target.checked)
              setComponentEnable(ce)
              animatorRef.current.setComponentEnable(ce)
            }}
            equation="A\sin(\phi + \omega x)"
          >
            {[
              ["A", "A"],
              ["omega", "\\omega"],
              ["phi", "\\phi"],
            ].map(([n, tn]) => (
              <ComponentParameter
                name={n}
                tex_name={tn}
                value={modelState.get(n)}
                posterior_value={posteriorState.get(n)}
                onChange={modelChange}
              ></ComponentParameter>
            ))}
          </ModelComponent>
        </div>
      </div>
      <div className="extra-components">
        empty posterior: <span id="empty-posterior">{emptyPosterior}</span>
        <br />
        <label>
          pause
          <input
            id="pause"
            type="checkbox"
            onChange={(e) => animatorRef.current.setPause(e.target.checked)}
          />
        </label>
        <label>
          Auto-SIR
          <input
            id="auto-SIR"
            type="checkbox"
            onChange={(e) => animatorRef.current.setAutoSIR(e.target.checked)}
          />
        </label>
      </div>
      <div className="card">
        <button id="sir" type="button" onClick={SIR_Update}>
          Update my priors, SIR!
        </button>
        <button id="reset-priors" type="button" onClick={Reset}>
          Reset
        </button>
      </div>
    </>
  )
}

function ModelComponent({
  name,
  enabled,
  onChange,
  equation = undefined,
  children,
}: {
  name: string
  enabled: boolean
  onChange: React.EventHandler<ChangeEvent<HTMLInputElement>>
  equation?: string
  children: React.ReactNode
}) {
  return (
    <div className="modelComponent">
      <div className="prior-component-title">
        <input
          className="model-component-enable"
          id={name + "_enable"}
          checked={enabled}
          onChange={onChange}
          type="checkbox"
        />
        {name}
        <div>
          <span className="katex-render" katex-source={equation}></span>
        </div>
      </div>
      {children}
    </div>
  )
}

function ComponentParameter({
  name,
  tex_name,
  value,
  posterior_value,
  onChange,
}) {
  const min = { mu: -2, sigma: 0 }
  const max = { mu: 2, sigma: 2 }
  const innerParams = ["mu", "sigma"].map((innerName) => {
    const joint_name = name + "_" + innerName
    return (
      <>
        <span
          className="katex-render"
          katex-source={
            tex_name + "\\hskip{0.5em}\\" + innerName + ":\\hskip{0.2em} "
          }
        ></span>
        <input
          type="range"
          min={min[innerName as keyof Distribution]}
          max={max[innerName as keyof Distribution]}
          step=".01"
          value={value[innerName]}
          id={joint_name}
          onChange={(e) => onChange(name, innerName, e.target.value)}
        />
        <span id={joint_name + "-value"}>
          {Number(value[innerName]).toFixed(2)}
        </span>
        &nbsp;&nbsp;
        <span className="posterior" id={joint_name + "-posterior"}>
          {Number(posterior_value[innerName]).toFixed(2)}
        </span>
        <br />
      </>
    )
  })

  return <div className="value-group">{innerParams}</div>
}

function InferenceUI({ K, N, setK, setN }) {
  const ns = [100, 1000, 5000, 10000, 50000, 100000].map((i) => (
    <option key={i} value={i}>
      {i.toLocaleString()}
    </option>
  ))
  const ks = [1, 5, 10, 25, 50].map((i) => (
    <option key={i} value={i}>
      {i.toLocaleString()}
    </option>
  ))
  return (
    <div id="inference-parameters">
      <label htmlFor="importanceSamplesPerParticle">N = \,</label>
      <select
        name="N"
        value={N}
        onChange={(e) => setN(parseInt(e.target.value))}
      >
        {ns}
      </select>
      &nbsp;&nbsp;
      <label htmlFor="numParticles">K =</label>
      <select
        id="numParticles"
        name="K"
        value={K}
        onChange={(e) => setK(e.target.value)}
      >
        {ks}
      </select>
    </div>
  )
}
