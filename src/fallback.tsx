import React from "react"


export const FALLBACK_MESSAGE = (error: Error) => (
  <React.Fragment>
    <div className="error-message">{error.toString()}</div>
    <div className="help-message">
      <p>
        <b>To enable WebGPU:</b>
      </p>
      <ul>
        <li>
          <b>Chrome 113+</b> – Windows, MacOS, ChromeOS ✅
        </li>
        <li>
          <b>Safari</b> – Go to Settings &rarr; Feature Flags and check "WebGPU"
        </li>
        <li>
          <b>Chrome</b> – Linux, Android - Dev version required
          <br />
          Turn on <code>#enable-unsafe-webgpu</code> in{" "}
          <code>chrome://flags</code>
        </li>
        <li>
          <b>Firefox</b> – Nightly version required
          <br />
          Turn on <code>dom.webgpu.enabled</code> in <code>about:config</code>
        </li>
      </ul>
      <p>
        Note that WebGPU requires an HTTPS connection if not running on{" "}
        <code>localhost</code>.
      </p>
      <p>
        See <a href="https://caniuse.com/webgpu">CanIUse.com</a> for more info.
      </p>
    </div>
  </React.Fragment>
)

//const F = () => {}
