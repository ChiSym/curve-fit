import { LC, PropsWithChildren, useOne, useState } from "@use-gpu/live"
import { Arrow, Line, Plot, Point, Transform } from "@use-gpu/plot"
import { HTML } from "@use-gpu/react"
import { AutoCanvas, WebGPU } from "@use-gpu/webgpu"
import {
  Data,
  DataShader,
  FlatCamera,
  getLineSegments,
  Kernel,
  LineLayer,
  Pass,
  PointLayer,
  useMouse,
  useWheel,
} from "@use-gpu/workbench"
import { FALLBACK_MESSAGE } from "../src/fallback"
import data from "./example_20_program.json"
import { LambdaSource, wgsl } from "@use-gpu/shader/wgsl"

type LocalizationViewProps = {
  canvas: HTMLCanvasElement
  fov: number
  nRays: number
}

export class Localization {
  walls_w: number
  walls_h: number
  walls: number[][]

  constructor() {
    this.walls = data.wall_verts
    const bbox = [
      [Infinity, Infinity],
      [-Infinity, -Infinity],
    ]
    this.walls.forEach(([x, y]) => {
      if (x < bbox[0][0]) bbox[0][0] = x
      if (x > bbox[1][0]) bbox[1][0] = x
      if (y < bbox[0][1]) bbox[0][1] = y
      if (y > bbox[1][1]) bbox[1][1] = y
    })
    this.walls_w = bbox[1][0] - bbox[0][0]
    this.walls_h = bbox[1][1] - bbox[0][1]
  }

  sensor_ps(angles: number[], p: number[], hd: number): number[][] {
    const ps = []
    for (const a of angles) {
      const a1 = a + hd
      const dp = [Math.cos(a1), Math.sin(a1)]
      const q = intersection_point(p, dp, this.walls)
      if (q) ps.push(q)
    }
    return ps
  }

  View: LC<LocalizationViewProps> = (
    props: PropsWithChildren<LocalizationViewProps>,
  ) => {
    const { canvas } = props
    return (
      <WebGPU
        fallback={(error: Error) => <HTML>{FALLBACK_MESSAGE(error)}</HTML>}
      >
        <AutoCanvas
          events
          backgroundColor={[0.9, 0.9, 0.9, 1.0]}
          canvas={canvas}
        >
          <FlatCamera>
            <Pass>
              <this.Viz {...props} />
            </Pass>
          </FlatCamera>
        </AutoCanvas>
      </WebGPU>
    )
  }

  Viz: LC<LocalizationViewProps> = (
    props: PropsWithChildren<LocalizationViewProps>,
  ) => {
    const { canvas, fov, nRays } = props
    const { mouse } = useMouse()
    const { wheel } = useWheel()

    const [xy, setXY] = useState<number[]>([0, 0])
    const [target, setTarget] = useState({ xy: [0, 0], hd: 0 })
    const [hd, setHd] = useState<number>(0)

    const box = canvas.parentElement!.getClientRects()[0]
    const inset = 0.1

    const scale_x = ((1 - 2 * inset) * box.width) / this.walls_w
    const scale_y = (-(1 - 2 * inset) * box.height) / this.walls_h
    const t_x = inset * box.width
    const t_y = (1 - inset) * box.height

    const angles = new Array(nRays)
    const fov_r = (fov / 180) * Math.PI
    for (let i = 0; i < nRays; ++i)
      angles[i] = (i / (nRays - 1)) * fov_r - fov_r / 2

    const matrix = [
      scale_x,
      0,
      0,
      0,
      0,
      scale_y,
      0,
      0,
      0,
      0,
      1,
      0,
      t_x,
      t_y,
      0,
      1,
    ]

    useOne(() => {
      const { x, y, buttons } = mouse

      // these coordinates are in window coordinates so we
      // need the matrix which transforms them to world
      // coordinates
      const x1 = (x - t_x) / scale_x
      const y1 = (y - t_y) / scale_y
      if (buttons.left) {
        setTarget({ xy: [x1, y1], hd: hd })
      }
      setXY([x1, y1])
    }, mouse)

    useOne(() => {
      setHd((hd) => (hd + wheel.moveY / 50) % (2 * Math.PI))
    }, wheel)

    const sps = this.sensor_ps(angles, xy, hd)
    const target_p2 = [
      target.xy[0] + Math.cos(target.hd),
      target.xy[1] + Math.sin(target.hd),
    ]

    const myShader = wgsl`
    @link fn getData(i: u32) -> vec2<f32>;

    fn main(i: u32) -> vec2<f32> {
      let s1 = getData(i);
      let s2 = getData(i+1);
      //return vec3<f32>((s1.x + s2.x) / 2, (s1.y + s2.y) / 2, 0);
      return vec2<f32>((s1.x + s2.x)/2, (s1.y + s2.y)/2);
      //return vec<f32>(max(0.0, sample), max(0.0, sample * .2) + max(0.0, -sample * .3), max(0.0, -sample), 1.0);
    }`;

    return (
      <Plot>
        <Transform
          /* position={[inset * box.width, (1 - inset) * box.height]}
        scale={[
          ((1 - 2 * inset) * box.width) / this.walls_w,
          (-(1 - 2 * inset) * box.height) / this.walls_h,
        ]} */ matrix={matrix}
        >
          <Data
            schema={{
              positions: "array<vec2<f32>>",
              colors: "vec4<f32>",
              widths: "f32",
              sizes: "f32",
            }}
            segments={getLineSegments}
            data={[
              {
                positions: data.wall_verts,
                colors: [0.7, 0, 0, 1],
                widths: 2,
                sizes: 100,
              },
            ]}
          >
            {(d) => { console.log(d); return (
              <>
                <LineLayer segments={d.segments} positions={d.positions} colors={d.colors} widths={d.widths} instances={d.instances}></LineLayer>
                <DataShader args={[xy]} shader={myShader} source={d.positions}>
                  {(midpoints: LambdaSource) => {
                    console.log(midpoints)
                    return <PointLayer positions={midpoints} size={d.size} color={[0,0,1,1]}></PointLayer>
                  }}
                </DataShader>
              </>
            )}}
          </Data>

          {/* <Line positions={this.walls}></Line> */}
          <Point color="#0f0" size={15} position={[0, 0]}></Point>
          <Point
            color="#00f"
            size={5}
            position={[this.walls_w, this.walls_h]}
          ></Point>
          <Point color="#880" size={4} positions={sps}></Point>
          <Arrow size={5} end positions={[target.xy, target_p2]}></Arrow>
          <Line positions={sps.map((q) => [xy, q])}></Line>
        </Transform>
      </Plot>
    )
  }
}

const PARALLEL_TOL = 1e-10
function solve_lines(p: number[], dp: number[], q: number[], dq: number[]) {
  const det = dp[0] * dq[1] - dp[1] * dq[0]
  if (Math.abs(det) < PARALLEL_TOL) return [-Infinity, -Infinity]
  return [
    (dq[0] * (p[1] - q[1]) - dq[1] * (p[0] - q[0])) / det,
    (dp[1] * (q[0] - p[0]) - dp[0] * (q[1] - p[1])) / det,
  ]
}

function distance(
  p: number[],
  dp: number[],
  q: number[],
  dq: number[],
): number {
  const [a, b] = solve_lines(p, dp, q, dq)
  if (a >= 0 && b >= 0 && b <= 1) return a
  return Infinity
}

function intersection_point(p: number[], dp: number[], walls: number[][]) {
  let min_d = Infinity
  for (let i = 0; i < walls.length - 1; ++i) {
    const d = distance(p, dp, walls[i], [
      walls[i + 1][0] - walls[i][0],
      walls[i + 1][1] - walls[i][1],
    ])
    if (d < min_d) {
      min_d = d
    }
  }
  return Number.isFinite(min_d) && [p[0] + min_d * dp[0], p[1] + min_d * dp[1]]
}
