
export class webgl {
  gl: WebGL2RenderingContext
  canvas: HTMLCanvasElement

  // TODO: wire this up
  constructor(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('webgl2')
    if (!ctx) throw new Error('Unable to create WebGL2 context')
    this.canvas = canvas
    this.gl = <WebGL2RenderingContext> ctx
  }

  private static createShader(gl: WebGL2RenderingContext, type: number, source: string): Promise<WebGLShader> {
    return new Promise((resolve, reject) => {
      const shader = gl.createShader(type)
      if (shader) {
        gl.shaderSource(shader, source)
        gl.compileShader(shader)
        if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
          resolve(shader)
        } else {
          const log = gl.getShaderInfoLog(shader)
          gl.deleteShader(shader)
          reject(log)
        }
      } else {
        reject('unable to create shader')
      }
    })
  }

  createProgram(vs: string, fs: string, varyings?: string[]): Promise<WebGLProgram> {
    return new Promise((resolve, reject) => {
      webgl.createShader(this.gl, this.gl.VERTEX_SHADER, vs).then(v_shader => {
        webgl.createShader(this.gl, this.gl.FRAGMENT_SHADER, fs).then(f_shader => {
          const program = this.gl.createProgram();
          if (program) {
            this.gl.attachShader(program, v_shader)
            this.gl.attachShader(program, f_shader)
            if (varyings) {
              this.gl.transformFeedbackVaryings(program, varyings, this.gl.SEPARATE_ATTRIBS)
            }
            this.gl.linkProgram(program)
            if (this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
              this.gl.deleteShader(v_shader)
              this.gl.deleteShader(f_shader)
              resolve(program)
            } else {
              const log = this.gl.getProgramInfoLog(program)
              this.gl.deleteProgram(program)
              reject(log)
            }
          } else {
            reject('unable to create program')
          }
        }, reason => {
          this.gl.deleteShader(v_shader)
          reject(reason)
        })
      }).catch(reject)
    })
  }

  makeBuffer(data: AllowSharedBufferSource | number) {
    const buf = this.gl.createBuffer()
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buf)
    if (typeof data === 'number') {
      // This is because bufferData has overloads, but the parameter is
      // declared as a union type
      this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW)
    } else {
      this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW)
    }
    return buf
  }

  uploadUints(buffer: WebGLBuffer, length: number): Uint32Array {
    const results = new Uint32Array(length)
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer)
    this.gl.getBufferSubData(this.gl.ARRAY_BUFFER, 0, results)
    return results
  }

  uploadFloats(buffer: WebGLBuffer, length: number): Float32Array {
    const results = new Float32Array(length);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer)
    this.gl.getBufferSubData(this.gl.ARRAY_BUFFER, 0, results)
    return results
  }


}
