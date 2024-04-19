import { MODEL_SIZE, Model, Normal } from "./model"
import { compute_shader } from "./shaders"
import { webgl } from "./webgl"

export type ResultBatch = {
  model: Float32Array,
  outliers: Uint32Array,
  weight: Float32Array,
  params: Float32Array
}

export type ModelParameters = {
  points: number[][],
  alpha: Normal[]
}

type InferenceResult = {
  selected_models: Model[],
  ips: number
}


export class GPGPU_Inference {
  private static UINTS_PER_SEED = 3
  private static PARAMS_SIZE = 3
  private static TF_BUFFER_NAMES = ['model', 'outliers', 'weight', 'params']
  private wgl: webgl
  private gl: WebGL2RenderingContext
  private program: WebGLProgram
  private pointsLoc: WebGLUniformLocation
  private alphaLocLoc: WebGLUniformLocation
  private alphaScaleLoc: WebGLUniformLocation
  private vao: WebGLVertexArrayObject
  private tfBuffers: Map<string, WebGLBuffer>
  // private modelBuffer: WebGLBuffer
  // private outliersBuffer: WebGLBuffer
  // private weightBuffer: WebGLBuffer
  // private paramsBuffer: WebGLBuffer
  private seedLoc: number
  private seedBuf: WebGLBuffer
  private tf: WebGLTransformFeedback
  private max_trials: number
  private seeds: Uint32Array
  private tfArrays: Map<string, Float32Array|Uint32Array>
  // private modelArray: Float32Array
  // private outliersArray: Uint32Array
  // private weightArray: Float32Array
  // private paramsArray: Float32Array

  constructor(max_trials: number) {
    const w = 2
    const h = 10
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const wgl = new webgl(canvas)
    const gl = wgl.gl

    const compute_fs = `#version 300 es
      precision highp float;
      void main() {
      }
    `

    this.vao = wgl.createVertexArray()

    const program = wgl.createProgram(compute_shader, compute_fs, GPGPU_Inference.TF_BUFFER_NAMES)

    this.seedLoc = gl.getAttribLocation(program, 'seed')
    // Create a VAO for the attribute state
    gl.bindVertexArray(this.vao)

    const makeBufferAndSetAttribute = (data: AllowSharedBufferSource, loc: number): WebGLBuffer => {
      const buf = wgl.makeBuffer(data)
      if (!buf) throw new Error('unable to create buffer')
      gl.enableVertexAttribArray(loc)
      gl.vertexAttribIPointer(
        loc,
        GPGPU_Inference.UINTS_PER_SEED,  // size (num components)
        gl.UNSIGNED_INT,
        0,
        0
      )
      return buf
    }

    this.max_trials = max_trials
    this.seeds = new Uint32Array(max_trials * GPGPU_Inference.UINTS_PER_SEED)
    this.seedBuf = makeBufferAndSetAttribute(this.seeds, this.seedLoc)

    // The vertex arrays (above) are for the INPUT.
    // The transform feedback (below) is for the OUTPUT.

    this.tf = wgl.createTransformFeedback()
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf)

    this.tfBuffers = new Map()
    this.tfBuffers.set('model', wgl.makeBuffer(max_trials * MODEL_SIZE * Float32Array.BYTES_PER_ELEMENT))
    this.tfBuffers.set('outliers', wgl.makeBuffer(max_trials * Uint32Array.BYTES_PER_ELEMENT))
    this.tfBuffers.set('weight', wgl.makeBuffer(max_trials * Float32Array.BYTES_PER_ELEMENT))
    this.tfBuffers.set('params', wgl.makeBuffer(max_trials * GPGPU_Inference.PARAMS_SIZE * Float32Array.BYTES_PER_ELEMENT))

    for (let [i, b] of GPGPU_Inference.TF_BUFFER_NAMES.entries()) gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, this.tfBuffers.get(b)!)

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null)
    gl.bindBuffer(gl.ARRAY_BUFFER, null)

    this.pointsLoc = wgl.getUniformLocation(program, 'points')
    this.alphaLocLoc = wgl.getUniformLocation(program, 'alpha_loc')
    this.alphaScaleLoc = wgl.getUniformLocation(program, 'alpha_scale')
    this.wgl = wgl
    this.gl = gl
    this.program = program

    // Pre-allocate buffers to receive the results of computation from the GPU.
    // We re-use these buffers on every call to `compute` to avoid generating
    // lots of garbage.
    this.tfArrays = new Map()
    this.tfArrays.set('model', new Float32Array(this.max_trials * MODEL_SIZE))
    this.tfArrays.set('outliers', new Uint32Array(this.max_trials))
    this.tfArrays.set('weight', new Float32Array(this.max_trials))
    this.tfArrays.set('params', new Float32Array(this.max_trials * GPGPU_Inference.PARAMS_SIZE))
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
    gl.uniform3f(this.alphaLocLoc, parameters.alpha[0].mu, parameters.alpha[1].mu, parameters.alpha[2].mu)
    gl.uniform3f(this.alphaScaleLoc, parameters.alpha[0].sigma, parameters.alpha[1].sigma, parameters.alpha[2].sigma)

    gl.bindVertexArray(this.vao)
    gl.enable(gl.RASTERIZER_DISCARD)
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf)
    gl.beginTransformFeedback(gl.POINTS)
    gl.drawArrays(gl.POINTS, 0, this.max_trials)
    gl.endTransformFeedback()
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null)
    gl.disable(gl.RASTERIZER_DISCARD)
    // END COMPUTE PART

    for (const b of GPGPU_Inference.TF_BUFFER_NAMES) {
      this.wgl.upload(this.tfBuffers.get(b)!, this.tfArrays.get(b)!)
    }

    // INSPECT RESULTS
    return {
      model: this.tfArrays.get('model') as Float32Array,
      outliers: this.tfArrays.get('outliers') as Uint32Array,
      weight: this.tfArrays.get('weight') as Float32Array,
      params: this.tfArrays.get('params') as Float32Array
    }
  }

  private logsumexp(a: Float32Array) {
    let sum_exp = 0.0;
    for (let i = 0; i < a.length; ++i) sum_exp += Math.exp(a[i])
    return Math.log(sum_exp)
  }

  private logit_to_probability(a: Float32Array) {
    const lse = this.logsumexp(a)
    for (let i = 0; i < a.length; ++i) a[i] = Math.exp(a[i] - lse)
  }

  inference(n_batches: number, parameters: ModelParameters): InferenceResult {
    let inference_time = 0.0
    let selected_models: Model[] = new Array(n_batches)
    const t0 = performance.now()
    for (let i = 0; i < n_batches; ++i) {
      const results: ResultBatch = this.compute(parameters)

      // now we need to look through the logit-indexed results table
      const weights = results.weight
      this.logit_to_probability(weights)
      const z = Math.random()
      // go thru the array until we have accumulated at least z's worth
      let target_index = 0
      let accumulated_prob = 0.0
      for (; target_index < weights.length; ++target_index) {
        accumulated_prob += weights[target_index];
        if (accumulated_prob >= z) break;
      }
      if (target_index >= weights.length) {
        // log('info', `oddly enough, the weights table ran out of probability for ${z} : ${accumulated_prob}`)
        // the above happens more often than I thought it would.
        // TODO: figure out why. theories:
        // 1) the model can generate a weight of NaN somehow
        // 2) all samples are so absurdly unlikely that the total probability
        //    content representable in double is 0.0, so that we cannot normalize
        target_index = weights.length - 1
      }
      selected_models[i] = {
        model: results.model.slice(target_index * MODEL_SIZE, target_index * MODEL_SIZE + MODEL_SIZE),
        outliers: results.outliers[target_index],
        weight: results.weight[target_index],
        params: results.params.slice(target_index * GPGPU_Inference.PARAMS_SIZE, target_index * GPGPU_Inference.PARAMS_SIZE + GPGPU_Inference.PARAMS_SIZE)
      }
    }
    inference_time += performance.now() - t0
    const ips = (n_batches * this.max_trials) / (inference_time/1e3)
    return {
      selected_models: selected_models,
      ips: ips
    }
  }


  cleanup() {
    this.gl.deleteBuffer(this.seedBuf)
    this.tfBuffers.forEach(this.gl.deleteBuffer)
  }
}
