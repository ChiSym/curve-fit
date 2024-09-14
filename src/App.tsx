import "./App.css"
import { Animator, InferenceReport } from "./animator.ts"
import { useCallback, useState, useRef, ChangeEvent } from "react"
import throttle from "lodash.throttle"
import katex from "katex"
import { InferenceParameters, InferenceResult } from "./gpgpu.ts"
import { RunningStats } from "./stats.ts"
import { TypedObject } from "./utils"
import { Component } from "./live.tsx"
import { LiveCanvas } from "@use-gpu/react"

class DistributionShape {
  public readonly name: string
  public readonly parameterName: string[]
  constructor(name: string, parameterName: string[]) {
    this.name = name
    this.parameterName = parameterName
  }
}

const NormalDistributionShape = new DistributionShape("normal", ["mu", "sigma"])

export class XDistribution {
  public readonly shape: DistributionShape
  public readonly parameters: Map<string, number>

  constructor(shape: DistributionShape, parameters: number[]) {
    this.shape = shape
    if (parameters.length != this.shape.parameterName.length) {
      throw new Error(
        `incorrect number of paramters for ${this.shape.name} distribution: ${parameters.length} != ${this.shape.parameterName.length}`,
      )
    }
    this.parameters = new Map()
    for (let i = 0; i < parameters.length; ++i) {
      this.parameters.set(this.shape.parameterName[i], parameters[i])
    }
  }
  public get(pName: string): number {
    return this.parameters.get(pName)!
  }
  public set(pName: string, value: number) {
    return this.parameters.set(pName, value)
  }
  public assoc(pName: string, value: number) {
    const ret = this.clone()
    ret.set(pName, value)
    return ret
  }
  public clone(): XDistribution {
    return new XDistribution(this.shape, Array.from(this.parameters.values()))
  }
  public getParameterNames(): string[] {
    return this.shape.parameterName
  }
}

export const Normal = (mu: number, sigma: number) =>
  new XDistribution(NormalDistributionShape, [mu, sigma])

export const modelParams: TypedObject<XDistribution> = {
  a_0: Normal(0, 2),
  a_1: Normal(0, 2),
  a_2: Normal(0, 2),
  omega: Normal(0, 2),
  A: Normal(0, 2),
  phi: Normal(0, 2),
  inlier: Normal(0.3, 0.07),
}

const defaultInferenceParameters: InferenceParameters = {
  importanceSamplesPerParticle: 1000,
  numParticles: 10,
}

export default function CurveFit() {
  const animatorRef = useRef<Animator | null>(null)

  const [inferenceParameters, setInferenceParameters] = useState(
    defaultInferenceParameters,
  )

  const [points, setPoints] = useState(() => {
    const xs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
    const points = xs.map((x) => [x, 0.7 * x + 0.3 * Math.sin(9.0 * x + 0.3)])
    points[7][1] = 0.75
    return { points: points, evictionIndex: 0 }
  })

  const [emptyPosterior, setEmptyPosterior] = useState(0)
  const [modelState, setModelState] =
    useState<TypedObject<XDistribution>>(modelParams)
  const [posteriorState, setPosteriorState] =
    useState<TypedObject<XDistribution>>(modelParams)

  const [componentEnable, setComponentEnable] = useState<TypedObject<boolean>>({
    polynomial: true,
    periodic: true,
  })

  const [ips, setIps] = useState(0.0)
  const [fps, setFps] = useState(0.0)
  const [visualizeInlierSigma, setVisualizeInlierSigma] = useState(false)

  function modelChange(k1: string, k2: string, v: number) {
    console.log(`${k1}_${k2} -> ${v}`)
    setModelState({ ...modelState, [k1]: modelState[k1]!.assoc(k2, v) })
    animatorRef.current?.setModelParameters(modelState)
  }

  const [outlier, setOutlier] = useState(Normal(0, 0))
  const setPOutlier = throttle((outlierStats: RunningStats) => {
    setOutlier(outlierStats.summarize())
  }, 250)

  const throttledSetIps = throttle(setIps, 500)
  const throttledSetFps = throttle(setFps, 500)
  const throttledSetPosteriorState = throttle(() => {
    setPosteriorState(animatorRef.current?.getPosterior() || {})
  }, 500)

  function SIR_Update() {
    const s = animatorRef.current?.getPosterior()
    if (s) {
      setModelState(s)
      setPosteriorState(s)
      animatorRef.current?.setModelParameters(s)
    }
  }

  const [inferenceResult, setInferenceResult] = useState<InferenceResult>({
    selectedModels: [],
    ips: 0,
    failedSamples: 0,
  })

  function setter(data: InferenceReport) {
    // This function is handed to the inference loop, which uses it to convey summary data
    // back to the UI.
    setEmptyPosterior(data.totalFailedSamples)
    throttledSetIps(data.inferenceResult.ips)
    throttledSetFps(data.fps)
    setPOutlier(data.pOutlierStats)
    if (data.autoSIR) {
      SIR_Update()
    } else {
      throttledSetPosteriorState()
    }
    setInferenceResult(data.inferenceResult)
  }

  const canvasRef = useCallback((canvas: HTMLCanvasElement | null) => {
    if (canvas) {
      const a = (animatorRef.current = new Animator(
        modelParams,
        defaultInferenceParameters,
        setter
      ))
      a.setInferenceParameters(inferenceParameters)
      a.setModelParameters(modelParams)
      a.setPoints(points.points)
      a.setComponentEnable(componentEnable)
      return a.run()
    }
  }, [])

  const componentsRef = useCallback((element: HTMLElement | null) => {
    if (element) {
      // Render all the spans tagged with TeX source with KaTeX
      Array.from(element.querySelectorAll("span.katex-render")).forEach((e) => {
        const text = e.getAttribute("katex-source")!
        katex.render(text, e as HTMLElement)
      })
    }
  }, [])

  function Reset() {
    setModelState(modelParams)
    setPosteriorState(modelParams)
    animatorRef.current?.setModelParameters(modelParams)
    animatorRef.current?.Reset()
  }

  function canvasClick(event: React.MouseEvent<HTMLElement>) {
    console.log("event.target", event.target)
    const canvas = event.target as HTMLCanvasElement
    const rect = canvas.getBoundingClientRect()
    console.log(rect)
    const x = ((event.clientX - rect.left) / rect.width) * 2.0 - 1.0
    const y = ((event.clientY - rect.top) / rect.height) * -2.0 + 1.0

    const ps = points.points
    let i = points.evictionIndex
    ps[i][0] = x
    ps[i][1] = y
    if (++i >= ps.length) {
      i = 0
    }
    setPoints({ points: ps.slice(), evictionIndex: i })
    animatorRef.current?.setPoints(ps)
    Reset()
  }

  return (
    <>
      <div className="live-canvas" onClick={canvasClick}>
        <LiveCanvas>
          {(canvas) => (
            <Component
              canvas={canvas}
              inferenceResult={inferenceResult}
              points={points.points}
              visualizeInlierSigma={visualizeInlierSigma}
            />
          )}
        </LiveCanvas>
      </div>
      <div className="manual-canvas">
        <canvas ref={canvasRef}></canvas>
      </div>
      <br />
      FPS: <span id="fps">{fps}</span>
      <br />
      IPS: <span id="ips">{(ips / 1e6).toFixed(2) + " M"}</span>
      <br />
      {(outlier.get("mu") || outlier.get("sigma")) && (
        <span id="outlier">
          p<sub>outlier</sub> = {outlier.get("mu").toFixed(2)} &plusmn;{" "}
          {outlier.get("sigma").toFixed(2)}
        </span>
      )}
      <InferenceUI
        K={inferenceParameters.numParticles}
        N={inferenceParameters.importanceSamplesPerParticle}
        setK={(K: number) => {
          const newIP = { ...inferenceParameters, numParticles: K }
          setInferenceParameters(newIP)
          animatorRef.current?.setInferenceParameters(newIP)
        }}
        setN={(N: number) => {
          const newIP = {
            ...inferenceParameters,
            importanceSamplesPerParticle: N,
          }
          setInferenceParameters(newIP)
          animatorRef.current?.setInferenceParameters(newIP)
        }}
      ></InferenceUI>
      <div ref={componentsRef} id="model-components">
        <div className="column">
          <ModelComponent
            name="polynomial"
            enabled={componentEnable.polynomial}
            onChange={(e) => {
              const ce = { ...componentEnable, polynomial: e.target.checked }
              setComponentEnable(ce)
              animatorRef.current?.setComponentEnable(ce)
            }}
            equation="a_0 + a_1 x + a_2 x^2"
          >
            {["a_0", "a_1", "a_2"].map((n) => (
              <ComponentParameter
                name={n}
                tex_name={n}
                value={modelState[n]}
                posterior_value={posteriorState[n]}
                onChange={modelChange}
              ></ComponentParameter>
            ))}
          </ModelComponent>
          <ModelComponent
            name="inlier sigma"
            enabled={null}
            onChange={() => null}
            equation=""
          >
            <ComponentParameter
              name="inlier"
              tex_name="\sigma_\mathrm{in}"
              value={modelState.inlier}
              posterior_value={posteriorState.inlier}
              onChange={modelChange}
            ></ComponentParameter>
          </ModelComponent>
        </div>
        <div className="column">
          <ModelComponent
            name="periodic"
            enabled={componentEnable.periodic}
            onChange={(e) => {
              const ce = { ...componentEnable, periodic: e.target.checked }
              setComponentEnable(ce)
              animatorRef.current?.setComponentEnable(ce)
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
                value={modelState[n]}
                posterior_value={posteriorState[n]}
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
          <input
            id="pause"
            type="checkbox"
            onChange={(e) => animatorRef.current?.setPause(e.target.checked)}
          />
          pause
        </label>
        &nbsp;&nbsp;
        <label>
          <input
            id="auto-SIR"
            type="checkbox"
            onChange={(e) => animatorRef.current?.setAutoSIR(e.target.checked)}
          />
          Auto-SIR
        </label>
        &nbsp;&nbsp;
        <label>
          <input
            id="visualizeInlierSigma"
            type="checkbox"
            onChange={(e) => setVisualizeInlierSigma(e.target.checked)}
          />
          viz inlier sigma
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
  enabled: boolean | null
  onChange: React.EventHandler<ChangeEvent<HTMLInputElement>>
  equation?: string
  children: React.ReactNode
}) {
  return (
    <div className="modelComponent">
      <div className="prior-component-title">
        {enabled !== null && (
          <input
            className="model-component-enable"
            id={name + "_enable"}
            checked={enabled}
            onChange={onChange}
            type="checkbox"
          />
        )}
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
}: {
  name: string
  tex_name: string
  value: XDistribution
  posterior_value: XDistribution
  onChange: (name: string, innerName: string, value: number) => void
}) {
  const min = { mu: -2, sigma: 0 }
  const max = { mu: 2, sigma: 2 }
  const innerParams = value.getParameterNames().map((innerName) => {
    const joint_name = name + "_" + innerName
    const keyName = innerName
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
          min={min[keyName as keyof typeof min]}
          max={max[keyName as keyof typeof min]}
          step={0.1}
          value={value.get(keyName)}
          id={joint_name}
          onChange={(e) => onChange(name, innerName, Number(e.target.value))}
        />
        <span id={joint_name + "-value"}>{value.get(keyName).toFixed(2)}</span>
        &nbsp;&nbsp;
        <span className="posterior" id={joint_name + "-posterior"}>
          {posterior_value.get(keyName).toFixed(2)}
        </span>
        <br />
      </>
    )
  })

  return <div className="value-group">{innerParams}</div>
}

function InferenceUI({
  K,
  N,
  setK,
  setN,
}: {
  K: number
  N: number
  setK: (k: number) => void
  setN: (n: number) => void
}) {
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
      <label htmlFor="importanceSamplesPerParticle">N =</label>
      <select
        id="importanceSamplesPerParticle"
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
        onChange={(e) => setK(parseInt(e.target.value))}
      >
        {ks}
      </select>
    </div>
  )
}
