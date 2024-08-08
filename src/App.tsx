import "./App.css"
import { Animator, Distribution } from "./main.ts"
import { useState, useEffect, useRef } from "react"
import throttle from "lodash.throttle"
import katex from "katex"
import { InferenceParameters } from "./gpgpu.ts"

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

  //const [modelParameters, setModelParameters] = useState([params.map(p => p.initialValue.mu), params.map(p=>p.initialValue.sigma))
  //  TODO: do we need an initial copy?
  const [emptyPosterior, setEmptyPosterior] = useState(0)
  const [modelState, setModelState] = useState(new Map(modelParams.entries()))
  const [posteriorState, setPosteriorState] = useState(
    new Map(modelParams.entries()),
  )

  const [ips, setIps] = useState(0.0)
  const [fps, setFps] = useState(0.0)

  // setInnerText("#fps", fps.toString())
  // setInnerText("#ips", `${(result.ips / 1e6).toFixed(1)} M`)

  function handleClick() {
    console.log("click")
  }

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

  function setter(data) {
    // This function is handed to the inference loop, which uses it to convey summary data
    // back to the UI.
    console.log("SETTING", data)
    setEmptyPosterior(data.totalFailedSamples)
    throttledSetIps(data.ips)
    throttledSetFps(data.fps)
    console.log("SPS", new Map(data.posterior))
    throttledSetPosteriorState(new Map(Array.from(data.posterior.entries()).map(([k, v]) => [k, v.summarize()])))
  }

  useEffect(() => {
    // Things to do once the UI is set up:
    // - Render all the spans tagged with TeX source with KaTeX
    Array.from(document.querySelectorAll("span.katex-render")).forEach((e) => {
      const text = e.getAttribute("katex-source")!
      console.log("renderin", text)
      katex.render(text, e as HTMLElement)
    })
    // - Start the animation loop
    animatorRef.current.setInferenceParameters(inferenceParameters)
    animatorRef.current.setModelParameters(modelParams)
    return animatorRef.current.run()
  }, [])

  return (
    <>
      <canvas id="c" onClick={handleClick}></canvas>
      <br />
      FPS: <span id="fps">{fps}</span>
      <br />
      IPS: <span id="ips">{Number(ips / 1e6).toFixed(2) + " M"}</span>
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
          <ModelComponent name="polynomial" equation="a_0 + a_1 x + a_2 x^2">
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
          <ModelComponent name="inlier (not working)" equation="">
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
          <ModelComponent name="periodic" equation="A\sin(\phi + \omega x)">
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
          <input id="pause" type="checkbox" />
        </label>
        <label>
          Auto-SIR
          <input id="auto-SIR" type="checkbox" />
        </label>
      </div>
    </>
  )
}

function ModelComponent({
  name,
  equation = undefined,
  children,
}: {
  name: string
  equation?: string
  children: React.ReactNode
}) {
  return (
    <div className="modelComponent">
      <div className="prior-component-title">
        <input
          className="model-component-enable"
          id={name + "_enable"}
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
  const innerParams = ["mu", "sigma"].map((innerName) => {
    const joint_name = name + "_" + innerName
    return (
      <>
        <span
          className="katex-render"
          katex-source={tex_name + "\\hskip{0.5em}\\" + innerName + ": "}
        ></span>
        <input
          type="range"
          min="-1"
          max="1"
          step=".01"
          defaultValue="0"
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
  // const [importanceSamplesPerParticle, setImportanceSamplesPerParticle] = useState(1000)
  // const [numParticles, setNumParticles] = useState(25)

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
