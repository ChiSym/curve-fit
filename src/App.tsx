import "./App.css"
import main from "./main.ts"
import { useState, useEffect, useRef } from "react"

import katex from "katex"

let started = false

export default function CurveFit() {
  function handleClick() {
    console.log("click")
    console.log("click c.props", c.props)
  }

  function onLoad(e) {
    console.log("loaded!", e)
  }

  function modelChange(k, v) {
    console.log(`${k} -> ${v}`)
  }

  useEffect(() => {
    console.log("useEffect!")
    if (!started) {
      started = true
      main()
    }
  }, [])

  return (
    <>
      <canvas id="c" onClick={handleClick} onLoad={onLoad}></canvas>
      <br />
      FPS: <span id="fps"></span>
      <br />
      IPS: <span id="ips"></span>
      <br />
      <InferenceParameters
        onChange={(k, v) => console.log(`${k} -> ${v}`)}
      ></InferenceParameters>
      <div id="model-components">
        <div className="column">
          <ModelComponent name="polynomial" equation="a_0 + a_1 x + a_2 x^2">
            {["a_0", "a_1", "a_2"].map((n) => (
              <ComponentParameter
                name={n}
                onChange={modelChange}
              ></ComponentParameter>
            ))}
          </ModelComponent>
          <ModelComponent name="inlier (not working)" equation="\cos(x)">
            <ComponentParameter
              name="inlier"
              onChange={modelChange}
            ></ComponentParameter>
          </ModelComponent>
        </div>
        <div className="column">
          <ModelComponent name="periodic" equation="A\sin(\phi + \omega x)">
            {["A", "omega", "phi"].map((n) => (
              <ComponentParameter
                name={n}
                onChange={modelChange}
              ></ComponentParameter>
            ))}
          </ModelComponent>
        </div>
      </div>
      <div className="extra-components">
        empty posterior: <span id="empty-posterior"></span>
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
  children: React.ReactElement[]
}) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    console.log("ACK")
    if (ref.current && equation) katex.render(equation, ref.current)
  }, [equation])

  return (
    <div className="modelComponent">
      <div className="prior-component-title">
        <input
          className="model-component-enable"
          id={name + "_enable"}
          type="checkbox"
        />
        {name}
        <div ref={ref}></div>
      </div>
      {children}
    </div>
  )
}

function ComponentParameter({ name, onChange }) {
  const nameMap = {
    a_0: (
      <span>
        a<sub>0</sub>
      </span>
    ),
    a_1: (
      <span>
        a<sub>1</sub>
      </span>
    ),
    a_2: (
      <span>
        a<sub>2</sub>
      </span>
    ),
    mu: <span>&mu;</span>,
    sigma: <span>&sigma;</span>,
    omega: <span>&omega;</span>,
    phi: <span>&phi;</span>,
  }

  // TODO: generate these from the pair

  const innerParams = ["mu", "sigma"].map((innerName) => {
    const joint_name = name + "_" + innerName
    return (
      <>
        <span>
          {nameMap[name as keyof typeof nameMap] || name}{" "}
          {nameMap[innerName as keyof typeof nameMap]}:
        </span>
        <input
          type="range"
          min="-1"
          max="1"
          step=".01"
          defaultValue="0"
          id={joint_name}
          onChange={(e) => onChange(joint_name, e.target.value)}
        />
        <span id={joint_name + "-value"}></span>
        &nbsp;&nbsp;
        <span className="posterior" id={joint_name + "-posterior"}></span>
        <br />
      </>
    )
  })

  return <div className="value-group">{innerParams}</div>
}

function InferenceParameters({ onChange }) {
  // const [importanceSamplesPerParticle, setImportanceSamplesPerParticle] = useState(1000)
  // const [numParticles, setNumParticles] = useState(25)

  const ns = [100, 1000, 5000, 10000, 50000, 100000].map((i) => (
    <option key={i} value={i}>
      {i.toLocaleString()}
    </option>
  ))
  console.log("ns", ns)
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
        defaultValue={1000}
        onChange={(e) => onChange("N", parseInt(e.target.value))}
      >
        {ns}
      </select>
      &nbsp;&nbsp;
      <label htmlFor="numParticles">K =</label>
      <select
        id="numParticles"
        name="K"
        defaultValue={10}
        onChange={(e) => onChange("K", parseInt(e.target.value))}
      >
        {ks}
      </select>
    </div>
  )
}
