import { ReactNode, StrictMode, useEffect } from "react"
import { createRoot } from "react-dom/client"
import { ErrorBoundary, FallbackProps } from "react-error-boundary"
import statLib from "./stat-lib.wgsl?raw"
import {
  GPULoadOp,
  GPUStoreOp,
} from "three/src/renderers/webgpu/utils/WebGPUConstants.js"

async function gpuDevice() {
  if (!navigator.gpu) throw new Error("no WebGPU support")
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error("Unable to attach WebGPU")
  return await adapter.requestDevice()
}

async function main() {
  const device = await gpuDevice()
  const sampleNormal = device.createShaderModule({
    label: "initial foray",
    code: /* wgsl */ `
      ${statLib}
      // =======

      @group(0)
      @binding(0)
      var <storage, read_write> data: array<f32>;

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
  const compInfo = await sampleNormal.getCompilationInfo()
  compInfo.messages.forEach((m) =>
    console.log(`${m.type} ${m.lineNum}:${m.linePos} ${m.message}`),
  )
  const N = 50000

  const t0 = performance.now()
  const pipeline = device.createComputePipeline({
    label: "initial pipeline",
    layout: "auto",
    compute: {
      module: sampleNormal,
    },
  })
  const output = new Float32Array(N)
  const workBuffer = device.createBuffer({
    label: "work buffer",
    size: output.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })
  // const resultBuffer = device.createBuffer({
  //   label: 'result buffer',
  //   size: output.byteLength,
  //   usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  // })
  const bindGroup = device.createBindGroup({
    label: "bind group for work buffer",
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: workBuffer } }],
  })
  const encoder = device.createCommandEncoder({
    label: "initial foray encoder",
  })
  const pass = encoder.beginComputePass({
    label: "initial foray pass",
  })
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(output.length)
  pass.end()

  // TODO: remove this when we get the shaders joined together
  //encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, resultBuffer.size)

  // ==================================================================
  const histo = device.createShaderModule({
    label: "histogram",
    code: `
      @group(0) @binding(0) var<storage, read_write> data: array<f32>;
      @group(0) @binding(1) var<uniform> u: Uniforms;
      @group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>>;

      struct Uniforms {
        left: f32,
        right: f32,
      };

      @compute
      @workgroup_size(1)
      fn compute(
        @builtin(global_invocation_id) id: vec3u
      ) {
        let i = id.x;
        let d: f32 = data[i];
        if (d < u.left || d > u.right) { return; }
        let w: f32 = u.right - u.left;
        let bin: u32 = u32((d - u.left) / w * f32(arrayLength(&bins)));
        atomicAdd(&bins[bin], 1u);
      }
    `,
  })
  histo
    .getCompilationInfo()
    .then((info) =>
      info.messages.forEach((m) =>
        console.log(`${m.type} ${m.lineNum}:${m.linePos} ${m.message}`),
      ),
    )
  const histoPipeline = device.createComputePipeline({
    label: "histo pipeline",
    layout: "auto",
    compute: {
      module: histo,
    },
  })
  console.log(histoPipeline)
  const M = 400
  const bins = new Uint32Array(M)
  const binBuffer = device.createBuffer({
    label: "bin buffer",
    size: bins.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })
  const uniformBuffer = device.createBuffer({
    size: 8, // 2 * f32,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  })
  new Float32Array(uniformBuffer.getMappedRange()).set([-3, 3])
  uniformBuffer.unmap()

  const binBindGroup = device.createBindGroup({
    label: "bind group for histogram",
    layout: histoPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: workBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: binBuffer } },
    ],
  })

  const pass2 = encoder.beginComputePass({ label: "histo pass" })
  pass2.setPipeline(histoPipeline)
  pass2.setBindGroup(0, binBindGroup)
  pass2.dispatchWorkgroups(N)
  pass2.end()

  // ==================================================================
  const canvas = document.querySelector<HTMLCanvasElement>("#normal-pdf")!
  const context = canvas?.getContext("webgpu")
  canvas.width = 400
  canvas.height = 400
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat()
  context?.configure({
    device,
    format: presentationFormat,
  })
  const render = device.createShaderModule({
    label: "render",
    code: `
      @vertex fn vs(
        @builtin(vertex_index) vertexIndex: u32
      ) -> @builtin(position) vec4f {
          let pos = array(
            vec2f(-1.0, -1.0),
            vec2f(1.0, -1.0),
            vec2f(-1.0, 1.0),
            vec2f(1.0, -1.0),
            vec2f(1.0, 1.0),
            vec2f(-1.0, 1.0),
          );
          return vec4f(pos[vertexIndex], 0.0, 1.0);
      }

      @group(0) @binding(0) var<storage, read_write> bins: array<u32>;

      @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
        let n: u32 = arrayLength(&bins);
        if (p.y < f32(n - bins[u32(p.x)])) {
          return vec4f(0.8,0.8,0.8,1.);
        } else {
          return vec4f(0.0, 0.0, 0.4, 0.0);
        }
      }

      // @group(0) @binding(0) var<storage, read_write> data: array<f32>;
      // @group(0) @binding(1) var<uniform> u: Uniforms;
      // @group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>>;

      // struct Uniforms {
      //   left: f32,
      //   right: f32,
      // };

      // @compute
      // @workgroup_size(1)
      // fn compute(
      //   @builtin(global_invocation_id) id: vec3u
      // ) {
      //   let i = id.x;
      //   let d: f32 = data[i];
      //   if (d < u.left || d > u.right) { return; }
      //   let w: f32 = u.right - u.left;
      //   let bin: u32 = u32((d - u.left) / w * f32(arrayLength(&bins)));
      //   atomicAdd(&bins[bin], 1u);
      // }
    `,
  })
  render
    .getCompilationInfo()
    .then((info) =>
      info.messages.forEach((m) =>
        console.log(`${m.type} ${m.lineNum}:${m.linePos} ${m.message}`),
      ),
    )
  const renderPipeline = device.createRenderPipeline({
    label: "render pipeline",
    layout: "auto",
    vertex: { module: render },
    fragment: {
      module: render,
      targets: [{ format: presentationFormat }],
    },
  })
  const renderPassDescriptor = {
    label: "render pass desc",
    colorAttachments: [
      {
        view: context!.getCurrentTexture().createView(),
        clearValue: [0.3, 0.3, 0.3, 1],
        loadOp: GPULoadOp.Clear,
        storeOp: GPUStoreOp.Store,
      },
    ],
  }

  const pass3 = encoder.beginRenderPass(renderPassDescriptor)

  const renderBindGroup = device.createBindGroup({
    label: "bind group for render",
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: binBuffer } }],
  })

  pass3.setPipeline(renderPipeline)
  pass3.setBindGroup(0, renderBindGroup)
  pass3.draw(6)
  pass3.end()

  // ==================================================================

  const binOutBuffer = device.createBuffer({
    label: "bin out buffer",
    size: bins.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  encoder.copyBufferToBuffer(binBuffer, 0, binOutBuffer, 0, binBuffer.size)

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  // await resultBuffer.mapAsync(GPUMapMode.READ)
  // const result = new Float32Array(resultBuffer.getMappedRange(0, resultBuffer.size))
  // const r = result.slice()
  // resultBuffer.unmap()

  await binOutBuffer.mapAsync(GPUMapMode.READ)
  const t1 = performance.now()
  console.log("perf", t1 - t0, (1000 * N) / (t1 - t0))
  const binOuts = new Uint32Array(
    binOutBuffer.getMappedRange(0, binOutBuffer.size),
  )
  const bs = binOuts.slice()
  binOutBuffer.unmap()
  const bTot = bs.reduce((a, b) => a + b)
  console.log("bs", bs)
  console.log("bTot", bTot)
  return bs
}

createRoot(document.getElementById("root")!, {
  onRecoverableError: (error, errorInfo) => {
    console.error("onRecoverableError", error, errorInfo)
  },
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
    console.log("effect")
    //main().then(x => setValue(x.join(', ')))
    main().then(() => {
      console.log("done")
    })
  })

  return (
    <div style={{ width: 400, height: 400 }}>
      <canvas
        id="normal-pdf"
        style={{ width: "100%", height: "100%" }}
      ></canvas>
    </div>
  )
}

function fallbackRender(fb: FallbackProps): ReactNode {
  return (
    <div className={"log-error"}>
      <pre>{fb.error.message}</pre>
    </div>
  )
}
