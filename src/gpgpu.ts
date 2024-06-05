import { MODEL_SIZE, type Model, type Normal } from "./model"
import { computeShader } from "./shaders"
import { WGL2Helper } from "./webgl"

export interface ResultBatch {
  model: Float32Array
  outliers: Uint32Array
  weight: Float32Array
  params: Float32Array
}

export interface ModelParameters {
  points: number[][]
  alpha: Normal[]
}

interface InferenceResult {
  selectedModels: Model[]
  ips: number
}

export class GPGPU_Inference {
  private static readonly UINTS_PER_SEED = 3
  private static readonly PARAMS_SIZE = 3
  private static readonly TF_BUFFER_NAMES = [
    "model",
    "outliers",
    "weight",
    "params",
  ]
  private readonly wgl: WGL2Helper
  private readonly gl: WebGL2RenderingContext
  private readonly program: WebGLProgram
  private readonly pointsLoc: WebGLUniformLocation
  private readonly alphaLocLoc: WebGLUniformLocation
  private readonly alphaScaleLoc: WebGLUniformLocation
  private readonly vao: WebGLVertexArrayObject
  private readonly tfBuffers: Map<string, WebGLBuffer>
  private readonly seedLoc: number
  private readonly seedBuf: WebGLBuffer
  private readonly tf: WebGLTransformFeedback
  private readonly max_trials: number
  private seeds: Uint32Array
  private readonly tfArrays: Map<string, Float32Array | Uint32Array>

  constructor(maxTrials: number) {
    const w = 2
    const h = 10
    const canvas = document.createElement("canvas")
    canvas.width = w
    canvas.height = h
    const wgl = new WGL2Helper(canvas)
    const gl = wgl.gl

    const computeFragmentShader = `#version 300 es
      precision highp float;
      void main() {
      }
    `

    this.vao = wgl.createVertexArray()

    const program = wgl.createProgram(
      computeShader,
      computeFragmentShader,
      GPGPU_Inference.TF_BUFFER_NAMES,
    )

    this.seedLoc = gl.getAttribLocation(program, "seed")
    // Create a VAO for the attribute state
    gl.bindVertexArray(this.vao)

    const makeBufferAndSetAttribute = (
      data: AllowSharedBufferSource,
      loc: number,
    ): WebGLBuffer => {
      const buf = wgl.makeBuffer(data)
      if (buf == null) throw new Error("unable to create buffer")
      gl.enableVertexAttribArray(loc)
      gl.vertexAttribIPointer(
        loc,
        GPGPU_Inference.UINTS_PER_SEED, // size (num components)
        gl.UNSIGNED_INT,
        0,
        0,
      )
      return buf
    }

    this.max_trials = maxTrials
    this.seeds = new Uint32Array(maxTrials * GPGPU_Inference.UINTS_PER_SEED)
    this.seedBuf = makeBufferAndSetAttribute(this.seeds, this.seedLoc)

    // The vertex arrays (above) are for the INPUT.
    // The transform feedback (below) is for the OUTPUT.

    this.tf = wgl.createTransformFeedback()
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf)

    this.tfBuffers = new Map()
    this.tfBuffers.set(
      "model",
      wgl.makeBuffer(maxTrials * MODEL_SIZE * Float32Array.BYTES_PER_ELEMENT),
    )
    this.tfBuffers.set(
      "outliers",
      wgl.makeBuffer(maxTrials * Uint32Array.BYTES_PER_ELEMENT),
    )
    this.tfBuffers.set(
      "weight",
      wgl.makeBuffer(maxTrials * Float32Array.BYTES_PER_ELEMENT),
    )
    this.tfBuffers.set(
      "params",
      wgl.makeBuffer(
        maxTrials *
          GPGPU_Inference.PARAMS_SIZE *
          Float32Array.BYTES_PER_ELEMENT,
      ),
    )

    for (const [i, b] of GPGPU_Inference.TF_BUFFER_NAMES.entries()) {
      const buf = this.tfBuffers.get(b)
      if (buf != null) {
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, buf)
      }
    }

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null)
    gl.bindBuffer(gl.ARRAY_BUFFER, null)

    this.pointsLoc = wgl.getUniformLocation(program, "points")
    this.alphaLocLoc = wgl.getUniformLocation(program, "alpha_loc")
    this.alphaScaleLoc = wgl.getUniformLocation(program, "alpha_scale")
    this.wgl = wgl
    this.gl = gl
    this.program = program

    // Pre-allocate buffers to receive the results of computation from the GPU.
    // We re-use these buffers on every call to `compute` to avoid generating
    // lots of garbage.
    this.tfArrays = new Map()
    this.tfArrays.set("model", new Float32Array(this.max_trials * MODEL_SIZE))
    this.tfArrays.set("outliers", new Uint32Array(this.max_trials))
    this.tfArrays.set("weight", new Float32Array(this.max_trials))
    this.tfArrays.set(
      "params",
      new Float32Array(this.max_trials * GPGPU_Inference.PARAMS_SIZE),
    )
  }

  // Runs the compute shader and retrieves the result of the computation.
  // NOTE that `compute` owns the storage holding the results, and that
  // storage will be overwritten on each call to compute.
  private compute(parameters: ModelParameters): ResultBatch {
    // DO THE COMPUTE PART
    const gl = this.gl
    gl.useProgram(this.program)

    // One GPU thread will be created for each _vertex_ in the array `seed`.
    // Each vertex is UINTS_PER_SEED unsigned 32 bit integers. The PRNG used
    // in the shader is designed to operate well with adjacent seeds, so we
    // generate one random integer and increment it to produce the seeds for
    // each thread.
    const baseSeed = Math.round(Math.random() * 0xffffffff)
    for (let i = 0; i < this.max_trials * GPGPU_Inference.UINTS_PER_SEED; ++i) {
      this.seeds[i] = baseSeed + i
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.seedBuf)
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.seeds)

    // points is a list: [[x1, y1], ...]
    // to send to the GPU, we flatten it: [x1, y1, x2, ...]
    gl.uniform2fv(this.pointsLoc, parameters.points.flat())
    gl.uniform3f(
      this.alphaLocLoc,
      parameters.alpha[0].mu,
      parameters.alpha[1].mu,
      parameters.alpha[2].mu,
    )
    gl.uniform3f(
      this.alphaScaleLoc,
      parameters.alpha[0].sigma,
      parameters.alpha[1].sigma,
      parameters.alpha[2].sigma,
    )

    gl.bindVertexArray(this.vao)
    gl.enable(gl.RASTERIZER_DISCARD)
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf)
    gl.beginTransformFeedback(gl.POINTS)
    gl.drawArrays(gl.POINTS, 0, this.max_trials)
    gl.endTransformFeedback()
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null)
    gl.disable(gl.RASTERIZER_DISCARD)
    // END COMPUTE PART

    for (const name of GPGPU_Inference.TF_BUFFER_NAMES) {
      const buf = this.tfBuffers.get(name)
      const arr = this.tfArrays.get(name)
      if (buf != null && arr != null) {
        this.wgl.upload(buf, arr)
      } else {
        throw new Error(`cannot find buffer/array ${name}`)
      }
    }

    // INSPECT RESULTS
    return {
      model: this.tfArrays.get("model") as Float32Array,
      outliers: this.tfArrays.get("outliers") as Uint32Array,
      weight: this.tfArrays.get("weight") as Float32Array,
      params: this.tfArrays.get("params") as Float32Array,
    }
  }

  private logsumexp(a: Float32Array): number {
    let sumExp = 0.0
    for (let i = 0; i < a.length; ++i) sumExp += Math.exp(a[i])
    return Math.log(sumExp)
  }

  private logit_to_probability(a: Float32Array): void {
    const lse = this.logsumexp(a)
    for (let i = 0; i < a.length; ++i) a[i] = Math.exp(a[i] - lse)
  }

  inference(nBatches: number, parameters: ModelParameters): InferenceResult {
    let inferenceTime = 0.0
    const selectedModels: Model[] = new Array(nBatches)
    const t0 = performance.now()
    for (let i = 0; i < nBatches; ++i) {
      const results: ResultBatch = this.compute(parameters)

      // now we need to look through the logit-indexed results table
      const weights = results.weight
      this.logit_to_probability(weights)
      const z = Math.random()
      // go thru the array until we have accumulated at least z's worth
      let targetIndex = 0
      let accumulatedProb = 0.0
      for (; targetIndex < weights.length; ++targetIndex) {
        accumulatedProb += weights[targetIndex]
        if (accumulatedProb >= z) break
      }
      if (targetIndex >= weights.length) {
        // log('info', `oddly enough, the weights table ran out of probability for ${z} : ${accumulated_prob}`)
        // the above happens more often than I thought it would.
        // TODO: figure out why. theories:
        // 1) the model can generate a weight of NaN somehow
        // 2) all samples are so absurdly unlikely that the total probability
        //    content representable in double is 0.0, so that we cannot normalize
        targetIndex = weights.length - 1
      }
      selectedModels[i] = {
        model: results.model.slice(
          targetIndex * MODEL_SIZE,
          targetIndex * MODEL_SIZE + MODEL_SIZE,
        ),
        outliers: results.outliers[targetIndex],
        weight: results.weight[targetIndex],
        params: results.params.slice(
          targetIndex * GPGPU_Inference.PARAMS_SIZE,
          targetIndex * GPGPU_Inference.PARAMS_SIZE +
            GPGPU_Inference.PARAMS_SIZE,
        ),
      }
    }
    inferenceTime += performance.now() - t0
    const ips = (nBatches * this.max_trials) / (inferenceTime / 1e3)
    return {
      selectedModels,
      ips,
    }
  }

  cleanup(): void {
    this.gl.deleteBuffer(this.seedBuf)
    this.tfBuffers.forEach((b) => {
      this.gl.deleteBuffer(b)
    })
  }
}
