import React from "react"
import { LiveCanvas } from "@use-gpu/react"
import { Localization } from "./loc-viz"

const localization = new Localization()

export default function Loc() {
  console.log("loc!")
  return (
    <React.Fragment>
      <h1>Loc</h1>

      <div className="live-canvas">
        <LiveCanvas>
          {(canvas) => <localization.View canvas={canvas} />}
        </LiveCanvas>
      </div>
    </React.Fragment>
  )
}
