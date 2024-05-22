export class WGL2Helper {
  gl: WebGL2RenderingContext
  canvas: HTMLCanvasElement

  constructor(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("webgl2")
    if (ctx == null) throw new Error("Unable to create WebGL2 context")
    this.canvas = canvas
    this.gl = ctx
  }

  getUniformLocation(
    program: WebGLProgram,
    name: string,
  ): WebGLUniformLocation {
    const l = this.gl.getUniformLocation(program, name)
    if (l == null) throw new Error(`unable to getUniformLocation(${name})`)
    return l
  }

  createVertexArray(): WebGLVertexArrayObject {
    const v = this.gl.createVertexArray()
    if (v == null) throw new Error("unable to createVertexArray()")
    return v
  }

  createTransformFeedback(): WebGLTransformFeedback {
    const tf = this.gl.createTransformFeedback()
    if (tf == null) throw new Error("unable to createTransformFeedback()")
    return tf
  }

  private createShader(type: number, source: string): WebGLShader {
    const gl = this.gl
    const shader = gl.createShader(type)
    if (shader != null) {
      gl.shaderSource(shader, source)
      gl.compileShader(shader)
      if (gl.getShaderParameter(shader, gl.COMPILE_STATUS) as GLboolean) {
        return shader
      } else {
        const log = gl.getShaderInfoLog(shader)
        gl.deleteShader(shader)
        throw new Error(log ?? "unknown shader creation error")
      }
    } else {
      throw new Error("unable to create shader")
    }
  }

  createProgram(vs: string, fs: string, varyings?: string[]): WebGLProgram {
    const vShader = this.createShader(this.gl.VERTEX_SHADER, vs)
    const fShader = this.createShader(this.gl.FRAGMENT_SHADER, fs)
    const program = this.gl.createProgram()
    if (program != null) {
      this.gl.attachShader(program, vShader)
      this.gl.attachShader(program, fShader)
      if (varyings != null) {
        this.gl.transformFeedbackVaryings(
          program,
          varyings,
          this.gl.SEPARATE_ATTRIBS,
        )
      }
      this.gl.linkProgram(program)
      if (this.gl.getProgramParameter(program, this.gl.LINK_STATUS) != null) {
        this.gl.deleteShader(vShader)
        this.gl.deleteShader(fShader)
        return program
      } else {
        const log = this.gl.getProgramInfoLog(program)
        this.gl.deleteProgram(program)
        throw new Error(log ?? "unknown WebGL2 program creation error")
      }
    } else {
      throw new Error("unable to create program")
    }
  }

  makeBuffer(data: AllowSharedBufferSource | number): WebGLBuffer {
    const buf = this.gl.createBuffer()
    if (buf == null) throw new Error("unable to createBuffer()")
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buf)
    if (typeof data === "number") {
      // This is because bufferData has overloads, but the parameter is
      // declared as a union type
      this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW)
    } else {
      this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW)
    }
    return buf
  }

  /**
   * upload the contents of the `buffer` from the device to the supplied
   * primitive array, which must be of sufficient size and correct type.
   * The previous content will be overwritten.
   */
  upload<T extends ArrayBufferView>(buffer: WebGLBuffer, out: T): void {
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer)
    this.gl.getBufferSubData(this.gl.ARRAY_BUFFER, 0, out)
  }
}
