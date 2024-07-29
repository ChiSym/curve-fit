import { WGL2Helper } from "./webgl.ts"
import { InferenceResult } from "./gpgpu.ts"

export class Render {
  private readonly positionLoc: number
  private readonly pointsLoc: WebGLUniformLocation
  private readonly canvasSizeLoc: WebGLUniformLocation
  private readonly nModelsLoc: WebGLUniformLocation
  private readonly coefficientsLoc: WebGLUniformLocation
  private readonly outliersLoc: WebGLUniformLocation
  private readonly gl: WebGL2RenderingContext
  private readonly program: WebGLProgram
  canvas: HTMLCanvasElement

  constructor(modelSize: number) {
    const c = document.querySelector<HTMLCanvasElement>("#c")
    if (c == null) throw new Error("unable to find canvas element for render")
    this.canvas = c
    this.canvas.width = 400
    this.canvas.height = 400
    const wgl = new WGL2Helper(this.canvas)
    const gl = wgl.gl

    const vs = `#version 300 es
    in vec4 a_position;
    void main() {
      gl_Position = a_position;
    }`

    const program = wgl.createProgram(vs, renderShader(modelSize))
    this.positionLoc = gl.getAttribLocation(program, "a_position")
    this.pointsLoc = wgl.getUniformLocation(program, "points")
    this.canvasSizeLoc = wgl.getUniformLocation(program, "canvas_size")
    this.nModelsLoc = wgl.getUniformLocation(program, "n_models")
    this.coefficientsLoc = wgl.getUniformLocation(program, "coefficients")
    this.outliersLoc = wgl.getUniformLocation(program, "outliers")

    // Set up full canvas clip space quad (this is two triangles that
    // together cover the space [-1,1] x [-1,1], the point being that
    // we want to run the fragment shader for every pixel in the "texture".)
    const buffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    )
    // Create a VAO for the attribute state
    const vao = gl.createVertexArray()
    gl.bindVertexArray(vao)
    // Tell WebGL how to pull data from the above array into
    // the position attribute of the vertex shader
    gl.enableVertexAttribArray(this.positionLoc)
    gl.vertexAttribPointer(
      this.positionLoc,
      2 /* count */,
      gl.FLOAT /* type */,
      false /* normalized */,
      0 /* stride */,
      0 /* offset */,
    )
    this.gl = gl
    this.program = program
  }

  render(points: number[][], result: InferenceResult): void {
    const gl = this.gl
    const models = result.selectedModels
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
    gl.useProgram(this.program)
    gl.uniform2f(this.canvasSizeLoc, gl.canvas.width, gl.canvas.height)
    gl.uniform2fv(this.pointsLoc, points.flat(), 0, 2 * points.length)
    gl.uniform1ui(this.nModelsLoc, result.selectedModels.length)
    gl.uniform1fv(
      this.coefficientsLoc,
      models.map((m) => Array.from(m.model)).flat(),
    )
    gl.uniform1uiv(
      this.outliersLoc,
      models.map((m) => m.outlier),
    )
    gl.clearColor(0.5, 0.5, 0.5, 1.0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }
}

function renderShader(modelSize: number): string {
  return /* glsl */ `#version 300 es
  precision highp float;
  #define N_POINTS 10
  #define MAX_N_MODELS 100u
  #define MODEL_SIZE ${modelSize}u
  #define M_PI 3.1415926535897932384626433832795

  uniform vec2 canvas_size;
  uniform uint n_models;
  uniform vec2 points[N_POINTS];
  uniform float coefficients[MAX_N_MODELS * MODEL_SIZE];
  uniform uint outliers[MAX_N_MODELS];

  out vec4 out_color;
  //uniform vec3 models[];

  void main() {
    // Map pixel coordinates [0,w) x [0,h) to the unit square [-1, 1) x [-1, 1)
    vec2 xy = gl_FragCoord.xy / canvas_size.xy * 2.0 + vec2(-1.0,-1.0);

    // background
    out_color = vec4(0.85,0.85,0.85,1.0);

    // rules
    if (mod(xy.x, 0.1) < 0.01 || mod(xy.y, 0.1) < 0.01) {
      out_color = vec4(0.9,0.9,1.0,1.0);
    }

    // axes
    if (abs(xy.x) < 0.006 || abs(xy.y) < 0.006) {
      out_color = (vec4(1.0,1.0,1.0,1.0));
    }

    // curves: blended
    for (uint i = 0u, ci = 0u; i < n_models; ++i, ci+=MODEL_SIZE) {
      float a0 = coefficients[ci];
      float a1 = coefficients[ci+1u];
      float a2 = coefficients[ci+2u];
      float omega = coefficients[ci+3u];
      float A = coefficients[ci+4u];
      float phi = coefficients[ci+5u];

      vec2 p;
      p.x = xy.x;
      p.y = a0 + p.x * a1 + p.x * p.x * a2;       // polynomial part
      p.y += A * sin(phi + omega * p.x); // periodic part

      // signed_distance to the implicit curve F(x, y) = 0 is
      //   F(x, y) / norm(grad(F)(x, y))
      // We have y = f(x), so F(x, y) = y - f(x)

      float Fxy = xy.y - p.y;
      vec2 GFxy = vec2(dFdx(Fxy), dFdy(Fxy));
      float sd = Fxy / length(GFxy);
      if (abs(sd) < 1.0) out_color = mix(vec4(0.0, 0.0, 0.0, 1.0), out_color, 0.65);
    }

    for (int i = 0; i < N_POINTS; ++i) {
      float d = distance(points[i], xy);
      // might need a smoothstep in here to antialias
      for (uint j = 0u; j < n_models; ++j) {
        if (d < 0.025) {
          // find out how many times this one is considered an outlier
          int outlier_count = 0;
          for (uint j = 0u; j < n_models; ++j) {
            if ((outliers[j] & (1u << i)) != 0u) {
              ++outlier_count;
            }
          }
          float outlier_frac = float(outlier_count) / float(n_models);
          out_color = outlier_frac * vec4(1.0,0.0,0.0,1.0) + (1.0-outlier_frac) * vec4(0.0,0.6,1.0,1.0);
        }
      }
    }
  }
  `;
}
