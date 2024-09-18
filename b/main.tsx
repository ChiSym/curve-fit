import { createRoot } from "react-dom/client"
import "./style.css"

import Loc from "./loc"
import { StrictMode } from "react"
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Loc />
  </StrictMode>,
)
