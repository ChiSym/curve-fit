import { ReactNode, StrictMode, useEffect, useRef, useState } from "react"
import { createRoot } from "react-dom/client"
import { ErrorBoundary, FallbackProps } from "react-error-boundary"
import statLib from "./stat-lib.wgsl?raw"
import {
  GPULoadOp,
  GPUStoreOp,
} from "three/src/renderers/webgpu/utils/WebGPUConstants.js"
import { FPSCounter } from "../src/fps_counter"
import throttle from "lodash.throttle"
import "./stat.css"

async function gpuDevice() {
  if (!navigator.gpu) throw new Error("no WebGPU support")
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error("Unable to attach WebGPU")
  return await adapter.requestDevice()
}

class ShaderApp {
  protected readonly device: GPUDevice

  constructor(device: GPUDevice) {
    this.device = device
  }

  public onClick() {}
  public build_pipeline(): () => void {
    return () => {
      throw Error("no pipeline")
    }
  }

  protected compile(label: string, code: string): GPUShaderModule {
    const m = this.device.createShaderModule({ label, code })
    m.getCompilationInfo().then((info) =>
      info.messages.forEach((msg) =>
        log(
          "error",
          `${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}`,
        ),
      ),
    )
    return m
  }
}
class Sampler extends ShaderApp {
  private readonly generate_normal: GPUShaderModule
  private readonly compute_histogram: GPUShaderModule
  private readonly render_histogram: GPUShaderModule
  private readonly presentationFormat: GPUTextureFormat
  private readonly context: GPUCanvasContext
  private readonly N: number
  private readonly M: number

  constructor(
    device: GPUDevice,
    canvasElementSelector: string,
    expression: string,
    N: number,
    M: number,
  ) {
    super(device)

    this.M = M
    this.N = N
    const canvas = document.querySelector<HTMLCanvasElement>(
      canvasElementSelector,
    )
    if (!canvas) throw Error("canvas not found")
    this.context = canvas.getContext("webgpu") as GPUCanvasContext
    if (!this.context) throw Error("no webgpu context")
    canvas.width = canvas.height = 400
    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat()
    this.context.configure({ device, format: this.presentationFormat })

    this.generate_normal = this.compile(
      "generate normal",
      `
      ${statLib}
      // =======

      @group(0) @binding(0) var <storage, read_write> data: array<f32>;
      @group(0) @binding(1) var <uniform> u: Uniforms;

      struct Uniforms {
        seed_bias: u32,
      };

      fn mixture(u: f32, m0: f32, s0: f32, m1: f32, s1: f32) -> f32 {
        let r = random_uniform(0.0, 1.0);
        if (r <= u) {
          return random_normal(m0, s0);
        } else {
          return random_normal(m1, s1);
        }
      }

      @compute
      @workgroup_size(1)
      fn compute(
        @builtin(global_invocation_id) id: vec3u
      ) {
        seed = id;
        seed.y = u.seed_bias + seed.y;
        let i = id.x;
        let u = ${expression};
        data[i] = u;
      }
    `,
    )

    this.compute_histogram = this.compile(
      "compute histogram",
      `
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
    )

    this.render_histogram = this.compile(
      "render histogram",
      `
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
        let b = bins[u32(p.x)];
        if (p.y < f32(n - b) && b <= n) {
          return vec4f(0.8,0.8,0.8,1.);
        } else {
          return vec4f(0.0, 0.0, 0.4, 0.0);
        }
      }
    `,
    )
  }

  public build_pipeline() {
    const generate = this.device.createComputePipeline({
      label: "generate pipeline",
      layout: "auto",
      compute: {
        module: this.generate_normal,
      },
    })
    const dataBuffer = this.device.createBuffer({
      label: "data buffer",
      size: this.N * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    })
    const uBuffer1 = this.device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      //mappedAtCreation: true,
    })
    //new Uint32Array(uBuffer1.getMappedRange()).set([1])
    //uBuffer1.unmap()
    const gBindGroup = this.device.createBindGroup({
      label: "generate bind group",
      layout: generate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dataBuffer } },
        { binding: 1, resource: { buffer: uBuffer1 } },
      ],
    })
    // ----------------------------------------------------------------
    const histCompute = this.device.createComputePipeline({
      label: "compute histogram",
      layout: "auto",
      compute: { module: this.compute_histogram },
    })
    const binBuffer = this.device.createBuffer({
      label: "bin buffer",
      size: this.M * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const uBuffer2 = this.device.createBuffer({
      size: 2 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    })
    new Float32Array(uBuffer2.getMappedRange()).set([-3, 3])
    uBuffer2.unmap()
    const binBindGroup = this.device.createBindGroup({
      label: "bind compute histogram",
      layout: histCompute.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dataBuffer } },
        { binding: 1, resource: { buffer: uBuffer2 } },
        { binding: 2, resource: { buffer: binBuffer } },
      ],
    })
    // ----------------------------------------------------------------
    const render = this.device.createRenderPipeline({
      label: "render histogram",
      layout: "auto",
      vertex: { module: this.render_histogram },
      fragment: {
        module: this.render_histogram,
        targets: [{ format: this.presentationFormat }],
      },
    })
    const renderBind = this.device.createBindGroup({
      label: "bind group for render",
      layout: render.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: binBuffer } }],
    })

    const colorAttachments = [
      {
        view: this.context.getCurrentTexture().createView(), // TODO: can we precompute this?
        clearValue: [0.3, 0.3, 0.3, 1],
        loadOp: GPULoadOp.Clear,
        storeOp: GPUStoreOp.Store,
      },
    ]

    const seed = new Uint32Array([10])

    return () => {
      const encoder = this.device.createCommandEncoder({
        label: "stat pipeline encoder",
      })
      const p1 = encoder.beginComputePass({ label: "generate pass" })
      p1.setPipeline(generate)
      p1.setBindGroup(0, gBindGroup)
      p1.dispatchWorkgroups(this.N)
      p1.end()
      this.device.queue.writeBuffer(uBuffer1, 0, seed)
      encoder.clearBuffer(binBuffer)
      const p2 = encoder.beginComputePass({ label: "histogram compute pass" })
      p2.setPipeline(histCompute)
      p2.setBindGroup(0, binBindGroup)
      p2.dispatchWorkgroups(this.N)
      p2.end()
      colorAttachments[0].view = this.context.getCurrentTexture().createView()
      const p3 = encoder.beginRenderPass({ colorAttachments })
      p3.setPipeline(render)
      p3.setBindGroup(0, renderBind)
      p3.draw(6)
      p3.end()
      seed.set([seed[0] + 1])
      this.device.queue.submit([encoder.finish()])
    }
  }
}

class GoL extends ShaderApp {
  private readonly generate_board: GPUShaderModule
  private readonly render_board: GPUShaderModule
  private readonly update_board: GPUShaderModule

  private readonly context: GPUCanvasContext
  private readonly presentationFormat: GPUTextureFormat
  private readonly size = 400
  private click = false
  public density = 0.3

  constructor(device: GPUDevice, canvasElementSelector: string) {
    super(device)

    const canvas = document.querySelector<HTMLCanvasElement>(
      canvasElementSelector,
    )
    if (!canvas) throw Error("canvas not found")
    this.context = canvas.getContext("webgpu") as GPUCanvasContext
    if (!this.context) throw Error("no webgpu context")
    canvas.width = canvas.height = 400
    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat()
    this.context.configure({ device, format: this.presentationFormat })

    this.generate_board = this.compile(
      "generate board",
      `
      ${statLib}
      // =======

      @group(0) @binding(0) var <storage, read_write> data: array<u32>;
      @group(0) @binding(1) var <uniform> u: Uniforms;

      struct Uniforms {
        seed_bias: u32,
        density: f32,
      };

      @compute
      @workgroup_size(1)
      fn compute(
        @builtin(global_invocation_id) id: vec3u
      ) {
        seed = id;
        seed.x = u.seed_bias + id.x;
        seed = pcg3d(seed);
        data[id.x * 400 + id.y] = u32(random_uniform(0.0, 1.0) < u.density);
      }
    `,
    )

    this.update_board = this.compile(
      "update board",
      `
      @group(0) @binding(0) var <storage, read_write> data0: array<u32>;
      @group(0) @binding(1) var <storage, read_write> data1: array<u32>;

      @compute
      @workgroup_size(1)
      fn compute(
        @builtin(global_invocation_id) id: vec3u
      ) {
        const stride = 400u;
        let i: u32 = id.x * stride + id.y;
        var n = 0u;
        if (id.x > 0 && data0[i-stride] != 0)        { n += 1; }    // N
        if (id.x < stride-1 && data0[i+stride] != 0) { n += 1; }    // S
        if (id.y > 0 && data0[i-1] != 0)             { n += 1; }    // W
        if (id.y < stride-1 && data0[i+1] != 0)      { n += 1; }    // E

        if (id.x > 0 && id.y > 0 && data0[i-stride-1] != 0) { n += 1; } // NW
        if (id.x < stride-1 && id.y > 0 && data0[i+stride-1] != 0) { n += 1; } // SW
        if (id.x > 0 && id.y < stride-1 && data0[i-stride+1] != 0) { n += 1; } // NE
        if (id.x < stride-1 && id.y < stride-1 && data0[i+stride+1] != 0) { n += 1; } // SE

        let c = data0[i];
        if (c == 0) {
          data1[i] = u32(n == 3);
        } else {
          data1[i] = u32(n == 2 || n == 3);
        }
      }
      `,
    )

    this.render_board = this.compile(
      "render board",
      `
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

      @group(0) @binding(0) var<storage, read_write> data: array<u32>;

      @fragment fn fs(
        @builtin(position) p: vec4f,
      ) -> @location(0) vec4f {

        if (data[u32(p.x - 0.5) * 400 + u32(p.y - 0.5)] != 0) {
          return vec4f(0,0,0,1);
        } else {
          return vec4f(0.8, 0.8, 0.8, 1.0);
        }
      }
    `,
    )
  }

  public onClick() {
    this.click = true
  }

  public build_pipeline() {
    const generate = this.device.createComputePipeline({
      label: "generate board",
      layout: "auto",
      compute: {
        module: this.generate_board,
      },
    })
    const dataBuffer0 = this.device.createBuffer({
      label: "data buffer 0",
      size: this.size * this.size * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    })
    const dataBuffer1 = this.device.createBuffer({
      label: "data buffer 1",
      size: this.size * this.size * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    })
    const uBuffer1 = this.device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT + Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const gBindGroup = this.device.createBindGroup({
      label: "generate bind group",
      layout: generate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dataBuffer0 } },
        { binding: 1, resource: { buffer: uBuffer1 } },
      ],
    })
    // ----------------------------------------------------------------
    const update = this.device.createComputePipeline({
      label: "update board",
      layout: "auto",
      compute: {
        module: this.update_board,
      },
    })
    const updateBindGroup0 = this.device.createBindGroup({
      label: "update bind group (even)",
      layout: update.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dataBuffer0 } },
        { binding: 1, resource: { buffer: dataBuffer1 } },
      ],
    })
    const updateBindGroup1 = this.device.createBindGroup({
      label: "update bind group (odd)",
      layout: update.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dataBuffer1 } },
        { binding: 1, resource: { buffer: dataBuffer0 } },
      ],
    })
    // ----------------------------------------------------------------
    const render = this.device.createRenderPipeline({
      label: "render board",
      layout: "auto",
      vertex: { module: this.render_board },
      fragment: {
        module: this.render_board,
        targets: [{ format: this.presentationFormat }],
      },
    })
    const renderBind0 = this.device.createBindGroup({
      label: "render bind (even)",
      layout: render.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: dataBuffer1 } }],
    })
    const renderBind1 = this.device.createBindGroup({
      label: "render bind (odd)",
      layout: render.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: dataBuffer0 } }],
    })
    // ----------------------------------------------------------------
    const colorAttachments = [
      {
        view: this.context.getCurrentTexture().createView(),
        clearValue: [0.3, 1.0, 0.3, 1],
        loadOp: GPULoadOp.Clear,
        storeOp: GPUStoreOp.Store,
      },
    ]

    const seed = new Uint32Array([10])
    const density = new Float32Array([this.density])

    // Initialize data buffer 0 with random data.

    const initializeBuffer = () => {
      seed.set([Math.floor(Math.random() * 0x7fffffff)])
      const encoder = this.device.createCommandEncoder({
        label: "GoL board initialize encoder",
      })
      const p1 = encoder.beginComputePass({ label: "generate pass" })
      p1.setPipeline(generate)
      p1.setBindGroup(0, gBindGroup)
      p1.dispatchWorkgroups(this.size, this.size)
      p1.end()
      this.device.queue.writeBuffer(uBuffer1, 0, seed)
      this.device.queue.writeBuffer(uBuffer1, 4, density)
      this.device.queue.submit([encoder.finish()])
    }

    initializeBuffer()

    let count = 0

    return () => {
      if (this.click) {
        initializeBuffer()
        this.click = false
        count = 0
      }
      const encoder = this.device.createCommandEncoder({
        label: "GoL board udpate encoder",
      })
      if (count > 1) {
        // allow initial data to show
        const p1 = encoder.beginComputePass({ label: "generate pass" })
        p1.setPipeline(update)
        p1.setBindGroup(0, count & 1 ? updateBindGroup1 : updateBindGroup0)
        p1.dispatchWorkgroups(this.size, this.size)
        p1.end()
      }
      colorAttachments[0].view = this.context.getCurrentTexture().createView()
      const p3 = encoder.beginRenderPass({ colorAttachments })
      p3.setBindGroup(0, count & 1 ? renderBind1 : renderBind0)
      p3.setPipeline(render)
      p3.draw(6)
      p3.end()
      this.device.queue.submit([encoder.finish()])
      count += 1
    }
  }
}

createRoot(document.getElementById("root")!, {
  onRecoverableError: (error, errorInfo) => {
    console.error("onRecoverableError", error, errorInfo)
  },
}).render(
  <StrictMode>
    <h1>distribution check</h1>
    <ErrorBoundary fallbackRender={fallbackRender}>
      <DistributionView
        elementId="normal-pdf"
        expression="random_normal(0.0, 1.0)"
      />
      <GoLView elementId="gol" />
      <DistributionView
        elementId="uniform-pdf"
        expression="random_uniform(-3.0, 3.0)"
      />
      <DistributionView
        elementId="mixture-pdf"
        expression="mixture(0.4, -1.0, 0.5, 1, 0.5)"
      />
    </ErrorBoundary>
  </StrictMode>,
)

function SetupAnimation(
  setFPS: (x: number) => void,
  ctor: (g: GPUDevice) => ShaderApp,
) {
  const fpsCounter = new FPSCounter()
  const throttledSetter = throttle((fps) => {
    setFPS(fps)
  }, 500)
  let stop = false
  gpuDevice()
    .then(ctor)
    .then((app) => {
      const draw = app.build_pipeline()

      requestAnimationFrame(function frame() {
        draw()
        throttledSetter(fpsCounter.observe())
        if (stop) {
          console.log("dropping animation frame cycle")
        } else {
          requestAnimationFrame(frame)
        }
      })
    })
  return () => {
    stop = true
  }
}

function GoLView({ elementId }: { elementId: string }) {
  const [fps, setFPS] = useState(0)
  const [sps, setSPS] = useState(0)
  const [density, setDensity] = useState(0)
  const appRef = useRef<ShaderApp>()
  useEffect(() => {
    return SetupAnimation(
      (fps) => {
        setFPS(fps)
        setSPS(400 * 400 * fps)
      },
      (d: GPUDevice) => {
        const app = (appRef.current = new GoL(d, "#" + elementId))
        setDensity((app.density = 0.3))
        return app
      },
    )
  }, [])

  function onClick() {
    appRef.current?.onClick()
  }

  return (
    <div>
      <div style={{ width: 400, height: 400 }}>
        <canvas
          onClick={onClick}
          id={elementId}
          style={{ width: "100%", height: "100%" }}
        ></canvas>
      </div>
      <div>
        <span className="expression">{`GoL(${density.toFixed(2)})`}</span>
        <span style={{ paddingLeft: "1em" }}>FPS: </span>
        <span>{fps}</span>
        <span style={{ paddingLeft: "1em" }}>Samples: </span>
        <span>{(sps / 1e6).toFixed(1) + "M/s"}</span>
        <span style={{ paddingLeft: "1em" }}>click to restart</span>
      </div>
    </div>
  )
}

function DistributionView({
  elementId,
  expression,
}: {
  elementId: string
  expression: string
}) {
  const [fps, setFPS] = useState(0)
  const [sps, setSPS] = useState(0)

  useEffect(() => {
    const N = 50000
    return SetupAnimation(
      (fps) => {
        setFPS(fps)
        setSPS(N * fps)
      },
      (d: GPUDevice) => new Sampler(d, "#" + elementId, expression, N, 400),
    )
  }, [])

  return (
    <div>
      <div style={{ width: 400, height: 400 }}>
        <canvas
          id={elementId}
          style={{ width: "100%", height: "100%" }}
        ></canvas>
      </div>
      <div>
        <span className="expression">{expression}</span>
        <span style={{ paddingLeft: "1em" }}>FPS: </span>
        <span>{fps}</span>
        <span style={{ paddingLeft: "1em" }}>Samples: </span>
        <span>{(sps / 1e6).toFixed(1) + "M/s"}</span>
      </div>
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

function log(level: string, message: unknown): void {
  if (level === "error") {
    console.error(message)
  } else {
    console.info(message)
  }
  const d = document.createElement("div")
  d.className = "log-" + level
  const t = document.createTextNode(message + "")
  d.appendChild(t)
  document.querySelector("#root")?.appendChild(d)
}
