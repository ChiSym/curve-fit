import type { Model } from "./model.ts"
import { WGL2Helper } from "./webgl.ts"

export class Render {
  private readonly positionLoc: number
  private readonly pointsLoc: WebGLUniformLocation
  private readonly canvasSizeLoc: WebGLUniformLocation
  private readonly nModelsLoc: WebGLUniformLocation
  private readonly polysLoc: WebGLUniformLocation
  private readonly outliersLoc: WebGLUniformLocation
  private readonly paramsLoc: WebGLUniformLocation
  private readonly gl: WebGL2RenderingContext
  private readonly program: WebGLProgram
  canvas: HTMLCanvasElement

  constructor() {
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

    const program = wgl.createProgram(vs, renderShader)
    this.positionLoc = gl.getAttribLocation(program, "a_position")
    this.pointsLoc = wgl.getUniformLocation(program, "points")
    this.canvasSizeLoc = wgl.getUniformLocation(program, "canvas_size")
    this.nModelsLoc = wgl.getUniformLocation(program, "n_models")
    this.polysLoc = wgl.getUniformLocation(program, "polys")
    this.outliersLoc = wgl.getUniformLocation(program, "outliers")
    this.paramsLoc = wgl.getUniformLocation(program, "params")

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

  render(points: number[][], models: Model[]): void {
    const gl = this.gl
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
    gl.useProgram(this.program)
    gl.uniform2f(this.canvasSizeLoc, gl.canvas.width, gl.canvas.height)
    gl.uniform2fv(this.pointsLoc, points.flat(), 0, 2 * points.length)
    gl.uniform1ui(this.nModelsLoc, models.length)
    gl.uniform3fv(this.polysLoc, models.map((m) => Array.from(m.model)).flat())
    gl.uniform1uiv(
      this.outliersLoc,
      models.map((m) => m.outliers),
    )
    gl.uniform3fv(
      this.paramsLoc,
      models.map((m) => Array.from(m.params)).flat(),
    )
    gl.clearColor(0.5, 0.5, 0.5, 1.0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }
}

const renderShader = /* glsl */ `#version 300 es
precision highp float;
#define N_POINTS 10
#define MAX_N_MODELS 100
uniform vec2 canvas_size;
uniform uint n_models;
uniform vec2 points[N_POINTS];
uniform vec3 polys[MAX_N_MODELS];
uniform uint outliers[MAX_N_MODELS];
uniform vec3 params[MAX_N_MODELS];

out vec4 out_color;
//uniform vec3 models[];

void main() {
  // Map pixel coordinates [0,w) x [0,h) to the unit square [-1, 1) x [-1, 1)
  vec2 xy = gl_FragCoord.xy / canvas_size.xy * 2.0 + vec2(-1.0,-1.0);

  for (int i = 0; i < N_POINTS; ++i) {

    float d = distance(points[i], xy);
    // might need a smoothstep in here to antialias
    if (d < 0.02) {
      // find out how many times this one is considered an outlier
      int outlier_count = 0;
      for (uint j = 0u; j < n_models; ++j) {
        if ((outliers[j] & (1u << i)) != 0u) {
          ++outlier_count;
        }
      }
      float outlier_frac = float(outlier_count) / float(n_models);
      out_color = outlier_frac * vec4(1.0,0.0,0.0,1.0) + (1.0-outlier_frac) * vec4(0.0,0.6,1.0,1.0);
      return;
    }
    out_color = vec4(0.0,0.5,0.0,0.8);
    uint circle_count = 0u;
    for (uint j = 0u; j < n_models; ++j) {
      if (abs(d - params[j].x) < 0.002) {
        out_color.g *= 0.8;
        ++circle_count;
      }
    }
    // if (circle_count > 0u) return;
  }

  uint curve_count = 0u;
  for (uint i = 0u; i < n_models; ++i) {
    vec3 poly = polys[i];
    vec2 p;
    p.x = xy.x;
    p.y = poly[0] + p.x * poly[1] + p.x * p.x * poly[2];
    float d = distance(p, xy);
    if (d < 0.01) {
      curve_count += 1u;
    }
  }
  if (curve_count > 0u) {
    float base = 0.7;
    float g = base * pow(0.8, float(curve_count));
    out_color = vec4(g,g,g,1.0);
  } else if (abs(xy.x) < 0.006 || abs(xy.y) < 0.006) {
    out_color = (vec4(1.0,1.0,1.0,1.0));
  } else if (mod(xy.x, 0.1) < 0.01 || mod(xy.y, 0.1) < 0.01) {
    out_color = vec4(0.9,0.9,1.0,1.0);
  } else {
    out_color = vec4(0.85,0.85,0.85,1.0);
  }
}
`
