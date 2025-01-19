import { ReactNode, StrictMode, useEffect, useState } from "react"
import { createRoot } from "react-dom/client"
import { ErrorBoundary, FallbackProps } from "react-error-boundary"

async function main() {

  const adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice()
  if (!device) {
    throw new Error("WebGPU not supported here")
  }
  const module = device.createShaderModule({
    label: "initial foray",
    code: /* wgsl */ `
      @group(0)
      @binding(0)
      var <storage, read_write> data: array<f32>;
      var <private> seed: vec3u;

      fn pcg3d(v: vec3u) -> vec3u {
        // Citation: Mark Jarzynski and Marc Olano, Hash Functions for GPU Rendering,
        // Journal of Computer Graphics Techniques (JCGT), vol. 9, no. 3, 21-38, 2020
        // Available online http://jcgt.org/published/0009/03/02/

        var w: vec3u = v * 1664525u + 1013904223u;
        w.x += w.y*w.z; w.y += w.z*w.x; w.z += w.x*w.y;
        w ^= w >> vec3u(16, 16, 16);
        w.x += w.y*w.z; w.y += w.z*w.x; w.z += w.x*w.y;
        return w;
      }

      // From Press NR 3ed.
      // A lower-order Chebyshev approximation produces a very concise routine, though with only about single precision accuracy:
      // Returns the complementary error function with fractional error everywhere less than 1.2e-7.
      fn erfc(x: f32) -> f32 {
        var z = abs(x);
        var t=2./(2.+z);
        var ans=t*exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
          t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
          t*(-0.82215223+t*0.17087277)))))))));
        return select(2.0 - ans, ans, x >= 0.0);
      }

      // The following two functions are from
      // http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function/
      fn inv_erfc(x: f32) -> f32 {
        let pp: f32 = select(2.0 - x, x, x < 1.0);
        let t: f32 = sqrt(-2.0 * log(pp/2.0));
        var r: f32;
        var er: f32;

        r = -0.70711 * ((2.30753 + t * 0.27061)/(1.0 + t * (0.99229 + t * 0.04481)) - t);
        er = erfc(r) - pp;
        r += er/(1.12837916709551257 * exp(-r * r) - r * er);
        //Comment the next two lines if you only wish to do a single refinement
        //err = erfc(r) - pp;
        //r += err/(1.12837916709551257 * exp(-r * r) - r * er);
        r = select(r, -r, x>1.0);
        return r;
      }

      fn inv_erf(x: f32) -> f32 {
        return inv_erfc(1.0-x);
      }

      fn random_normal(loc: f32, scale: f32) -> f32 {
        let u = sqrt(2.0) * inv_erf(random_uniform(-1.0, 1.0));
        return loc + scale * u;
      }

      // recovered from de-compiled JAX
      fn random_uniform(low: f32, high: f32) -> f32 {
        seed = pcg3d(seed);
        let a: f32 = bitcast<f32>((seed.x >> 9u) | 1065353216u) - 1.0;
        let diff = high - low;
        let w = diff * a;
        let u = w + low;
        return max(low, u);
      }

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

  const pipeline = device.createComputePipeline({
    label: 'initial pipeline',
    layout: 'auto',
    compute: {
      module,
    },
  })
  const input = new Float32Array([4,5,7])
  const workBuffer = device.createBuffer({
    label: 'work buffer',
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(workBuffer, 0, input);
  const resultBuffer = device.createBuffer({
    label: 'result buffer',
    size: input.byteLength,
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
  pass.dispatchWorkgroups(input.length)
  pass.end()
  encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, resultBuffer.size)
  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])
  await resultBuffer.mapAsync(GPUMapMode.READ)
  const result = new Float32Array(resultBuffer.getMappedRange(0, resultBuffer.size))
  const rs = result.toString()
  console.log(input)
  console.log(rs)
  resultBuffer.unmap()
  return rs
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
    main().then(x => setValue(x.toString()))
  })
  const [value, setValue] = useState("")

  return <span>This is a thing {value}</span>
}

function fallbackRender(fb: FallbackProps): ReactNode {
  return (
    <div className={"log-error"}>
      <pre>{fb.error.message}</pre>
    </div>
  )
}
