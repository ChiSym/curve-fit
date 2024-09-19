import React, { useState } from "react"
import { LiveCanvas } from "@use-gpu/react"
import { Localization } from "./loc-viz"

const localization = new Localization()

export default function Loc() {
  const [fov, setFov] = useState(240)
  const [nRays, setNRays] = useState(15)
  return (
    <React.Fragment>
      <h1>Loc</h1>

      <div className="live-canvas">
        <LiveCanvas>
          {(canvas) => <localization.View canvas={canvas} fov={fov} nRays={nRays}/>}
        </LiveCanvas>
      </div>
      <div>
        <input value={fov} id='fov' type='range' min='30' max='360' onChange={e => setFov(e.target.valueAsNumber)}></input>
        <label htmlFor="fov">FOV</label><br/>
        <input value={nRays} id='nRays' type='range' min='2' max='50' onChange={e => setNRays(e.target.valueAsNumber)}/>
        <label htmlFor="nRays">nRays</label>
      </div>
    </React.Fragment>
  )
}
