import { ReactNode, StrictMode, useEffect, useState } from "react"
import { createRoot } from "react-dom/client"
import { ErrorBoundary, FallbackProps } from "react-error-boundary"
import statLib from './stat-lib.wgsl?raw'

async function main() {

  console.log(statLib)
  const adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice()
  if (!device) {
    throw new Error("WebGPU not supported here")
  }
  const module = device.createShaderModule({
    label: "initial foray",
    code: /* wgsl */ `
      ${statLib}
      // =======

      @group(0)
      @binding(0)
      var <storage, read_write> data: array<f32>;
      var <private> seed: vec3u;

      @compute
      @workgroup_size(1)
      fn compute(
        @builtin(global_invocation_id) id: vec3u
      ) {
        seed = id;
        let i = id.x;
        let u = random_normal(0.0, 1.0);
        data[i] = u;
      }
    `,
  })
  const compInfo = await module.getCompilationInfo()
  compInfo.messages.forEach(m => console.log(`${m.type} ${m.lineNum}:${m.linePos} ${m.message}`))
  const N = 1000

  const pipeline = device.createComputePipeline({
    label: 'initial pipeline',
    layout: 'auto',
    compute: {
      module,
    },
  })
  const output = new Float32Array(N)
  const workBuffer = device.createBuffer({
    label: 'work buffer',
    size: output.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })
  const resultBuffer = device.createBuffer({
    label: 'result buffer',
    size: output.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })
  const bindGroup = device.createBindGroup({
    label: 'bind group for work buffer',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: workBuffer }},
    ]
  })
  const encoder = device.createCommandEncoder({
    label: 'initial foray encoder',
  })
  const pass = encoder.beginComputePass({
    label: 'initial foray pass',
  })
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(output.length)
  pass.end()
  encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, resultBuffer.size)
  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])
  await resultBuffer.mapAsync(GPUMapMode.READ)
  const result = new Float32Array(resultBuffer.getMappedRange(0, resultBuffer.size))
  const r = result.slice()
  resultBuffer.unmap()
  return r
}

createRoot(document.getElementById("root")!,
{
  onRecoverableError: (error, errorInfo) => {
    console.error('onRecoverableError', error, errorInfo)
  }
}).render(
  <StrictMode>
    <h1>this is from react</h1>
    <ErrorBoundary fallbackRender={fallbackRender}>
      <Thing />
    </ErrorBoundary>
  </StrictMode>,
)

function Thing() {
  useEffect(() => {
    main().then(x => setValue(x.join(', ')))
  })
  const [value, setValue] = useState("")

  return <div>This is a thing {value}</div>
}

function fallbackRender(fb: FallbackProps): ReactNode {
  return (
    <div className={"log-error"}>
      <pre>{fb.error.message}</pre>
    </div>
  )
}
