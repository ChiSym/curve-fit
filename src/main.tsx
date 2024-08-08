import { createRoot } from "react-dom/client"
import "./style.css"

import App from "./App"
import React from "react"
import { StrictMode } from "react"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <h1>SIR Curve Fit</h1>
    <App />
  </StrictMode>,
)
