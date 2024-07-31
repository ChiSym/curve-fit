import { type Model, type Normal } from "./model"
import { importanceShader } from "./shaders"
import { WGL2Helper } from "./webgl"

export interface ResultBatch {
  model: Float32Array
  weight: Float32Array
  p_outlier: Float32Array
  outlier: Uint32Array
}

export interface ModelParameters {
  points: number[][]
  coefficients: Normal[]
  component_enable: Map<string, boolean>
}

export interface InferenceParameters {
  numParticles: number
  importanceSamplesPerParticle: number
}

export interface InferenceResult {
  selectedModels: Model[]
  ips: number
  failedSamples: number
}

export class GPGPU_Inference {
  private static readonly UINTS_PER_SEED = 3
  private static readonly TF_BUFFER_NAMES = [
    "out_0",
    "out_1",
    "out_2",
    "out_3",
    "out_4",
    "out_5",
    "out_6",
    "out_log_weight",
    "out_p_outlier",
    "out_outliers",
    "out_inlier_sigma",
  ]
  private readonly wgl: WGL2Helper
  private readonly importanceProgram: WebGLProgram
  private readonly pointsLoc: WebGLUniformLocation
  private readonly alphaLocLoc: WebGLUniformLocation
  private readonly alphaScaleLoc: WebGLUniformLocation
  private readonly componentEnableLoc: WebGLUniformLocation
  private readonly vao: WebGLVertexArrayObject
  private readonly seedLoc: number
  private readonly seedBuf: WebGLBuffer
  private readonly tf: WebGLTransformFeedback
  private readonly max_trials: number
  private readonly nParameters: number
  private readonly floatsPerSeed: number
  private readonly seeds: Uint32Array
  private readonly weight: Float32Array
  private readonly parameters: Float32Array
  private readonly p_outlier: Float32Array
  private readonly outlier: Uint32Array

  private readonly bigArray: Float32Array
  private readonly bigBuf: WebGLBuffer

  constructor(nParameters: number, maxTrials: number) {
    const w = 2
    const h = 10
    const canvas = document.createElement("canvas")
    canvas.width = w
    canvas.height = h
    this.wgl = new WGL2Helper(canvas)
    const gl = this.wgl.gl

    const computeFragmentShader = `#version 300 es
      precision highp float;
      void main() {
      }
    `

    this.vao = this.wgl.createVertexArray()

    this.importanceProgram = this.wgl.createProgram(
      importanceShader(nParameters),
      computeFragmentShader,
      GPGPU_Inference.TF_BUFFER_NAMES,
    )
    this.seedLoc = gl.getAttribLocation(this.importanceProgram, "seed")
    // Create a VAO for the attribute state
    gl.bindVertexArray(this.vao)

    const makeBufferAndSetAttribute = (
      data: AllowSharedBufferSource,
      loc: number,
    ): WebGLBuffer => {
      const buf = this.wgl.makeBuffer(data)
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
    this.nParameters = nParameters
    this.seeds = new Uint32Array(maxTrials * GPGPU_Inference.UINTS_PER_SEED)
    // preallocate buffers to receive results of GPU computation
    this.weight = new Float32Array(this.max_trials)
    this.parameters = new Float32Array(this.max_trials * nParameters)
    this.p_outlier = new Float32Array(this.max_trials)
    this.outlier = new Uint32Array(this.max_trials)
    this.seedBuf = makeBufferAndSetAttribute(this.seeds, this.seedLoc)
    this.floatsPerSeed = GPGPU_Inference.TF_BUFFER_NAMES.length
    this.bigArray = new Float32Array(this.max_trials * this.floatsPerSeed)

    // The vertex arrays (above) are for the INPUT.
    // The transform feedback (below) is for the OUTPUT.
    this.tf = this.wgl.createTransformFeedback()
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf)

    this.bigBuf = this.wgl.makeBuffer(
      maxTrials * this.floatsPerSeed * Float32Array.BYTES_PER_ELEMENT,
    )
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.bigBuf)
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null)
    gl.bindBuffer(gl.ARRAY_BUFFER, null)

    this.pointsLoc = this.wgl.getUniformLocation(
      this.importanceProgram,
      "points",
    )
    this.alphaLocLoc = this.wgl.getUniformLocation(
      this.importanceProgram,
      "alpha_loc",
    )
    this.alphaScaleLoc = this.wgl.getUniformLocation(
      this.importanceProgram,
      "alpha_scale",
    )
    this.componentEnableLoc = this.wgl.getUniformLocation(
      this.importanceProgram,
      "component_enable",
    )
  }

  private sendParameters(parameters: ModelParameters) {
    const gl = this.wgl.gl
    // points is a list: [[x1, y1], ...]
    // to send to the GPU, we flatten it: [x1, y1, x2, ...]

    gl.uniform2fv(this.pointsLoc, parameters.points.flat())

    gl.uniform1fv(
      this.alphaLocLoc,
      parameters.coefficients.map((c) => c.mu),
    )
    gl.uniform1fv(
      this.alphaScaleLoc,
      parameters.coefficients.map((c) => c.sigma),
    )

    let enableBits = 0
    if (parameters.component_enable.get("polynomial")) enableBits |= 1
    if (parameters.component_enable.get("periodic")) enableBits |= 2

    gl.uniform1ui(this.componentEnableLoc, enableBits)
  }

  // Runs the compute shader and retrieves the result of the computation.
  // NOTE that `compute` owns the storage holding the results, and that
  // storage will be overwritten on each call to compute.
  private compute(
    parameters: ModelParameters,
    inferenceParameters: InferenceParameters,
  ): ResultBatch {
    // DO THE COMPUTE PART
    const N = inferenceParameters.importanceSamplesPerParticle
    const gl = this.wgl.gl
    gl.useProgram(this.importanceProgram)

    // One GPU thread will be created for each _vertex_ in the array `seed`.
    // Each vertex is UINTS_PER_SEED unsigned 32 bit integers. The PRNG used
    // in the shader is designed to operate well with adjacent seeds, so we
    // generate one random integer and increment it to produce the seeds for
    // each thread.
    const baseSeed = Math.round(Math.random() * 0xffffffff)
    for (let i = 0; i < N * GPGPU_Inference.UINTS_PER_SEED; ++i) {
      this.seeds[i] = baseSeed + i
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.seedBuf)
    gl.bufferSubData(
      gl.ARRAY_BUFFER,
      0,
      this.seeds,
      0,
      N * GPGPU_Inference.UINTS_PER_SEED,
    )

    this.sendParameters(parameters)

    gl.bindVertexArray(this.vao)
    gl.enable(gl.RASTERIZER_DISCARD)
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf)
    gl.beginTransformFeedback(gl.POINTS)
    gl.drawArrays(gl.POINTS, 0, N)
    gl.endTransformFeedback()
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null)
    gl.disable(gl.RASTERIZER_DISCARD)
    // END COMPUTE PART

    // Get the data back: interleaved mode

    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.bigBuf)
    gl.getBufferSubData(
      gl.TRANSFORM_FEEDBACK_BUFFER,
      0,
      this.bigArray,
      0,
      N * this.floatsPerSeed,
    )
    const a = this.bigArray
    for (let i = 0, p = 0; i < N; ++i, p += this.floatsPerSeed) {
      const m_i = i * this.nParameters
      this.parameters.set(a.subarray(p, p + this.nParameters), m_i)
      this.weight[i] = a[p + this.nParameters]
      this.p_outlier[i] = a[p + this.nParameters + 1]
      this.outlier[i] = a[p + this.nParameters + 2]
    }
    // TODO: unbind

    // INSPECT RESULTS
    return {
      model: this.parameters.subarray(0, N * this.floatsPerSeed),
      p_outlier: this.p_outlier.subarray(0, N),
      outlier: this.outlier.subarray(0, N),
      weight: this.weight.subarray(0, N),
    }
  }

  private logsumexp(a: Float32Array): number {
    let sumExp = 0.0
    a.forEach((v) => (sumExp += Math.exp(v)))
    return Math.log(sumExp)
  }

  private normalize(a: Float32Array): void {
    const lse = this.logsumexp(a)
    for (let i = 0; i < a.length; ++i) a[i] = Math.exp(a[i] - lse)
  }

  inference(
    modelParameters: ModelParameters,
    inferenceParameters: InferenceParameters,
  ): InferenceResult {
    let inferenceTime = 0.0
    let failedSamples = 0
    let modelIndex = 0
    const selectedModels: Model[] = new Array(inferenceParameters.numParticles)
    const t0 = performance.now()
    for (let i = 0; i < inferenceParameters.numParticles; ++i) {
      const results: ResultBatch = this.compute(
        modelParameters,
        inferenceParameters,
      )
      const weights = results.weight
      this.normalize(weights)
      const z = Math.random()
      // go thru the array until we have accumulated at least z's worth
      let targetIndex = 0
      let accumulatedProb = 0.0
      for (; targetIndex < weights.length; ++targetIndex) {
        accumulatedProb += weights[targetIndex]
        if (accumulatedProb >= z) break
      }
      if (targetIndex < weights.length) {
        selectedModels[modelIndex++] = {
          model: results.model.slice(
            targetIndex * this.nParameters,
            targetIndex * this.nParameters + this.nParameters,
          ),
          outlier: results.outlier[targetIndex],
          p_outlier: results.p_outlier[targetIndex],
          log_weight: results.weight[targetIndex],
        }
      } else {
        // log('info', `oddly enough, the weights table ran out of probability for ${z} : ${accumulated_prob}`)
        // the above happens more often than I thought it would.
        // TODO: figure out why. theories:
        // 1) the model can generate a weight of NaN somehow
        // 2) all samples are so absurdly unlikely that the total probability
        //    content representable in double is 0.0, so that we cannot normalize
        ++failedSamples
      }
    }
    inferenceTime += performance.now() - t0
    const ips =
      (inferenceParameters.numParticles *
        inferenceParameters.importanceSamplesPerParticle) /
      (inferenceTime / 1e3)
    return {
      selectedModels: selectedModels.slice(0, modelIndex),
      ips,
      failedSamples,
    }
  }

  private drift(models: Model[], mParams: ModelParameters, driftScale: number) {
    // prototype of drift in JS before shader implementation

    function random_normal() {
      const u1 = 1 - Math.random()
      const u2 = Math.random()
      const mag = Math.sqrt(-2 * Math.log(u1))
      return mag * Math.cos(2 * Math.PI * u2)
    }

    function evaluate(coefficients: Float32Array, x: number) {
      const y_poly =
        coefficients[0] + x * coefficients[1] + x * x * coefficients[2]
      const y_periodic =
        coefficients[4] + Math.sin(coefficients[5] + coefficients[3] * x)
      return y_poly + y_periodic
    }

    const xs = mParams.points.map((p) => p[0])

    for (const m of models) {
      const parameters = m.model
      const drifted = parameters.map((v) => v + driftScale * random_normal())
      const drifted_ys = xs.map((x) => evaluate(drifted, x))
    }
  }

  cleanup(): void {
    this.wgl.deleteBuffer(this.seedBuf)
    this.wgl.deleteBuffer(this.bigBuf)
  }
}
