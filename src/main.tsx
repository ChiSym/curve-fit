import { createRoot } from "react-dom/client"
import { ErrorBoundary, FallbackProps } from "react-error-boundary"
import "./style.css"
import version from "virtual:version"

import App from "./App"
import { ReactNode, StrictMode } from "react"
createRoot(document.getElementById("root")!, {
  onRecoverableError: (error, errorInfo) => {
    console.error("onRecoverableError", error, errorInfo)
  },
}).render(
  <StrictMode>
    <h1>SIR Curve Fit</h1>
    <ErrorBoundary fallbackRender={fallbackRender}>
      <App />
      <footer>{version}</footer>
    </ErrorBoundary>
  </StrictMode>,
)

function fallbackRender(fb: FallbackProps): ReactNode {
  return (
    <div className={"log-error"}>
      <pre>{fb.error.message}</pre>
    </div>
  )
}
