import './style.css'
import typescriptLogo from './typescript.svg'
import viteLogo from '/vite.svg'
import { setupCounter } from './counter.ts'
import { webgl } from './webgl.ts'
import { compute_shader, render_shader } from './shaders.ts'

document.querySelector<HTMLImageElement>('#vitelogo')!.src = viteLogo
document.querySelector<HTMLImageElement>('#tslogo')!.src = typescriptLogo

setupCounter(document.querySelector<HTMLButtonElement>('#counter')!)

function log(level: string, message: any): void {
  if (level == 'error') {
    console.error(message)
  } else {
    console.info(message)
  }
  const d = document.createElement('div')
  d.className = 'log-' + level
  const t = document.createTextNode(message.toString())
  d.appendChild(t)
  document.querySelector('#app')?.appendChild(d)
}

function uniform(low: number, high: number): number {
  const r = Math.random()
  return low + (high - low) * r
}

function rectangle(gl: WebGL2RenderingContext, x0: number, y0: number, x1: number, y1: number) {
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    x0, y0,
    x1, y0,
    x0, y1,
    x0, y1,
    x1, y0,
    x1, y1
  ]), gl.STATIC_DRAW)
  gl.drawArrays(gl.TRIANGLES, 0, 6)
}

// use the texture hack to sample from a distribution on the GPU
// use the texture hack to sample from a distribution on the GPU


// genjax intro interpreted data

// xs = jnp.array([0.3, 0.7, 1.1, 1.4, 2.3, 2.5, 3.0, 4.0, 5.0])
// ys = jnp.array(2.0 * xs + 1.5 + xs**2)
// ys = ys.at[2].set(50.0)

// TODO: find a less nested way to do this




async function boxes() {
  const vs = `#version 300 es
    in vec4 a_position;
    uniform mat4 u_matrix;
    void main() {
      gl_Position = u_matrix * a_position;
    }`

  const fs = `#version 300 es
    // fragment shaders don't have a default precision so we need
    // to pick one. highp is a good default. It means "high precision"
    precision highp float;

    uniform vec4 u_color;

    // we need to declare an output for the fragment shader
    out vec4 outColor;

    void main() {
      outColor = u_color;
    }`
  const wgl: webgl = new webgl(document.querySelector('#c')!)
  const gl = wgl.gl
  const program = await wgl.createProgram(vs, fs)
  const colorLocation = gl.getUniformLocation(program, 'u_color')
  const matrixLocation = gl.getUniformLocation(program, 'u_matrix')

  // looking up attribute/uniform locations is something to do during setup, not render loop
  const positionAttributeLocation = gl.getAttribLocation(program, "a_position")
  // attributes get their data from buffers
  const positionBuffer = gl.createBuffer()
  // first you bind a resource to a bind point. then all other fns refer to the resource thru the bind point. bind the position buffer
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
  // use the bind point to put data in the buffer


  // ======



  // it used the positionBuffer because we bound it to ARRAY_BUFFER above.
  // Now that we have the data in the buffer we need to tell the attribute how
  // to get data out of it.
  const vao = gl.createVertexArray()
  // make that vertex array current
  gl.bindVertexArray(vao)
  // turn the attribute on, so webgl will get data out of a buffer, else it will have constant value
  gl.enableVertexAttribArray(positionAttributeLocation)
  gl.vertexAttribPointer(
    positionAttributeLocation,
    2, /* size */
    gl.FLOAT, /* type */
    false, /* normalize */
    0, /* stride */
    0, /* offset */
  )

  // NB: the current ARRAY_BUFFER is bound to the attribute; i.e., the
  // attribute is bound to positionBuffer. This means we are now free to
  // rebind the ARRAY_BUFFER bind point. The attribute will carry on using
  // positionBuffer from now on.

  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
  gl.enable(gl.BLEND)
  gl.clearColor(0, 0, 0, 0)
  gl.clear(gl.COLOR_BUFFER_BIT)
  gl.useProgram(program)
  // ====== SET UP SCENE
  // const positions = [0,0, 0,0.5, 0.7, 0]
  // gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW)
  gl.uniformMatrix4fv(matrixLocation, true, [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ])
  const draw = () => uniform(-1, 1)
  for (let i = 0; i < 100; ++i) {
    let a = draw(), b = draw(), c = draw(), d = draw();
    gl.uniform4f(colorLocation, Math.random(), Math.random(), Math.random(), .9)
    rectangle(gl, a, b, c, d)
  }
}

const MODEL_SIZE = 3


async function gpgpu(n_trials: number) {
  const w = 2
  const h = 10
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const wgl = new webgl(canvas)
  const gl = wgl.gl

  log('info', `A ${gl.canvas.width} x ${gl.canvas.height}`)

  const compute_fs = `#version 300 es
    precision highp float;
    void main() {
    }
  `

  const program = await wgl.createProgram(compute_shader, compute_fs, [
    'model',
    'outliers',
    'weight'
  ])

  const aLoc = gl.getAttribLocation(program, 'a')
  // Create a VAO for the attribute state
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao)

  const makeBufferAndSetAttribute = (data: AllowSharedBufferSource, loc: number): WebGLBuffer => {
    const buf = wgl.makeBuffer(data)
    if (!buf) throw new Error('unable to create buffer')
    gl.enableVertexAttribArray(loc)
    gl.vertexAttribPointer(
      loc,
      1,  // size (num components)
      gl.FLOAT,
      false,
      0,
      0
    )
    return buf
  }

  // One GPU thread will be created for each element in the array `a`.
  // The array is filled with the integers [0..n_trials), so each
  // thread will receive a different integer in its a value; these
  // can be used as PRNG seeds
  const a = Array.from({ length: n_trials }, (_, i) => i)
  const aBuf = makeBufferAndSetAttribute(new Float32Array(a), aLoc)

  // The vertex arrays (above) are for the INPUT.
  // The transform feedback (below) is for the OUTPUT.

  const tf = gl.createTransformFeedback()
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf)

  const modelBuffer = wgl.makeBuffer(a.length * MODEL_SIZE * Float32Array.BYTES_PER_ELEMENT)
  const outliersBuffer = wgl.makeBuffer(a.length * Uint32Array.BYTES_PER_ELEMENT)
  const weightBuffer = wgl.makeBuffer(a.length * Float32Array.BYTES_PER_ELEMENT)

  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, modelBuffer)
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 1, outliersBuffer)
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 2, weightBuffer)

  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null)
  gl.bindBuffer(gl.ARRAY_BUFFER, null)

  const xsLoc = gl.getUniformLocation(program, 'xs')
  const ysLoc = gl.getUniformLocation(program, 'ys')

  const computation = () => {
    // DO THE COMPUTE PART
    gl.useProgram(program)


    // Send `xs` array to GPU
    const xs = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4]
    gl.uniform1fv(xsLoc, xs, 0, 10)
    // send `ys` array to GPU
    const ys = [.1, 50, .1, .2, .7, .3, .2, .1, .9, .3]
    gl.uniform1fv(ysLoc, xs, 0, 10)

    gl.bindVertexArray(vao)
    gl.enable(gl.RASTERIZER_DISCARD)
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf)
    gl.beginTransformFeedback(gl.POINTS)
    gl.drawArrays(gl.POINTS, 0, a.length)
    gl.endTransformFeedback()
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null)
    gl.disable(gl.RASTERIZER_DISCARD)
    // END COMPUTE PART

    // INSPECT RESULTS
    return {
      model: wgl.uploadFloats(modelBuffer!, a.length * MODEL_SIZE),
      outliers: wgl.uploadUints(outliersBuffer!, a.length),
      weight: wgl.uploadFloats(weightBuffer!, a.length)
    }
  }

  const cleanup = () => {
    gl.deleteBuffer(aBuf)
    gl.deleteBuffer(modelBuffer)
    gl.deleteBuffer(outliersBuffer)
    gl.deleteBuffer(weightBuffer)
  }

  return Promise.resolve({
    computation: computation,
    cleanup: cleanup
  })
}

async function render(points) {
  const canvas = document.createElement('canvas')
  canvas.width = 300
  canvas.height = 300
  document.querySelector('#app')?.appendChild(canvas)
  const wgl = new webgl(canvas)
  const gl = wgl.gl
  gl.getContextAttributes

  const vs = `#version 300 es
  in vec4 a_position;
  void main() {
    gl_Position = a_position;
  }`

  const program = await wgl.createProgram(vs, render_shader)
  const positionLoc = gl.getAttribLocation(program, 'a_position')!
  const pointsLoc = gl.getUniformLocation(program, 'points')
  //gl.uniform2fv(pointsLoc, points, 0, 10 * 2)
  const canvasSizeLoc = gl.getUniformLocation(program, 'canvas_size')

  // Set up full canvas clip space quad (this is two triangles that
  // together cover the space [-1,1] x [-1,1], the point being that
  // we want to run the fragment shader for every pixel in the "texture".)
  const buffer = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,
    1, -1,
    -1, 1,
    -1, 1,
    1, -1,
    1, 1]), gl.STATIC_DRAW)
  // Create a VAO for the attribute state
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao)
  // Tell WebGL how to pull data from the above array into
  // the position attribute of the vertex shader
  gl.enableVertexAttribArray(positionLoc)
  gl.vertexAttribPointer(
    positionLoc,
    2, /* count */
    gl.FLOAT, /* type */
    false, /* normalized */
    0, /* stride */
    0 /* offset */
  )



  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
  gl.useProgram(program)
  log('info', `ren canv ${canvas.width}, ${canvas.height}`)
  gl.uniform2f(canvasSizeLoc, canvas.width, canvas.height)
  log('info', program)
  log('info', `dw buffer ${gl.drawingBufferWidth}, ${gl.drawingBufferHeight}`)

  gl.clearColor(0.5,0.5,0.5,1.0)
  gl.clear(gl.COLOR_BUFFER_BIT)
  gl.drawArrays(gl.TRIANGLES, 0, 6)
  return Promise.resolve(true)
}

const N_BATCHES = 20

function logsumexp(a: Float32Array) {
  let sum_exp = 0.0;
  for (let i = 0; i < a.length; ++i) sum_exp += Math.exp(a[i])
  return Math.log(sum_exp)
}

function logit_to_probability(a: Float32Array) {
  const lse = logsumexp(a)
  for (let i = 0; i < a.length; ++i) a[i] = Math.exp(a[i] - lse)
}

const  N_INFERENCES_PER_BATCH = 10000
try {
  await boxes()
  const t0 = performance.now()
  const o = await gpgpu(N_INFERENCES_PER_BATCH)
  const t1 = performance.now()

  let results = Array(N_BATCHES)
  for (let i = 0; i < N_BATCHES; ++i) {
    results[i] = o.computation()
  }
  o.cleanup()
  const t2 = performance.now()
  const if_ps = Math.round((N_BATCHES * N_INFERENCES_PER_BATCH) / ((t2-t1)/1e3))
  console.log(`completed ${N_BATCHES} batches of ${N_INFERENCES_PER_BATCH} ${if_ps}/s}`)
  console.log('build time', t1 - t0)
  console.log('generate time', t2 - t1)
  console.log('total time', t2 - t0)
  const t3 = performance.now()
  let selected_models = new Float32Array(N_BATCHES * MODEL_SIZE)
  let sm_index = 0
  for (let i = 0; i < N_BATCHES; ++i) {
    // now we need to look through the logit-indexed results table
    const weights = results[i].weight
    logit_to_probability(weights)
    // let prob_sum = 0.0;
    // for (let i = 0; i < weights.length; ++i) prob_sum += weights[i]
    // console.log(`batch ${i} prob sum ${prob_sum} (diff from 1 ${1-prob_sum})`)
    const z = Math.random()
    // go thru the array until we have accumulated at least z's worth
    let target_index = 0
    let accumulated_prob = 0.0
    for (; target_index < weights.length; ++target_index) {
      accumulated_prob += weights[target_index];
      if (accumulated_prob >= z) break;
    }
    if (target_index >= weights.length) {
      log('info', `oddly enough, the weights table ran out of probability for ${z} : ${accumulated_prob}`)
      target_index = weights.length - 1
    }
    const target_model = results[i].model.slice(target_index, target_index+MODEL_SIZE)
    //console.log(`z = ${z}, index = ${target_index}, model = ${target_model}, outliers = ${results[i].outliers[target_index]}`)
    for (let j = 0; j < MODEL_SIZE; j++) selected_models[sm_index++] = target_model[j]
  }
  const t4 = performance.now()
  console.log('normalize time', t4 - t3)
  await render([[0.0,0.0],[0.0,0.5],[-0.5,-0.5]])
} catch (error) {
  log('error', error)
}
