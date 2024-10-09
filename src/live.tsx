import type { LC, PropsWithChildren } from "@use-gpu/live"
import { into } from "@use-gpu/live"
import { AutoCanvas, WebGPU } from "@use-gpu/webgpu"
import { HTML } from "@use-gpu/react"
import { FlatCamera, PanControls, Pass } from "@use-gpu/workbench"
import { FALLBACK_MESSAGE } from "./fallback"
import {
  Axis,
  Grid,
  Line,
  Plot,
  Point,
  Polygon,
  Transform,
} from "@use-gpu/plot"
import { InferenceResult } from "./gpgpu"

type ComponentProps = {
  canvas: HTMLCanvasElement
  inferenceResult: InferenceResult
  visualizeInlierSigma: boolean
  points: number[][]
}

const xs = new Array(60)
const h = 2.0 / (xs.length - 1)
for (let i = 0, x = -1; i < xs.length; ++i, x += h) xs[i] = x

export const Component: LC<ComponentProps> = (
  props: PropsWithChildren<ComponentProps>,
) => {
  const { canvas, inferenceResult, points, visualizeInlierSigma } = props
  function modelToFn(model: Float32Array) {
    return (x: number) => {
      const [a0, a1, a2, omega, A, phi] = model
      return a0 + x * a1 + x * x * a2 + A * Math.sin(phi + omega * x)
    }
  }
  const pts: number[][][] = inferenceResult.selectedModels.map((m) => {
    const f = modelToFn(m.model)
    return xs.map((x) => [x, f(x)])
  })
  const outlierCounts = new Array(points.length).fill(0)
  inferenceResult.selectedModels.forEach((m) => {
    for (let b = 1, i = 0; i < points.length; ++i, b <<= 1) {
      if ((m.outlier & b) != 0) {
        ++outlierCounts[i]
      }
    }
  })
  const colors = outlierCounts.map((c) => {
    const k = c / inferenceResult.selectedModels.length
    return [k, 0, 1 - k, 1]
  })
  return (
    <WebGPU
      fallback={(error: Error) => <HTML>{into(FALLBACK_MESSAGE(error))}</HTML>}
    >
      <AutoCanvas backgroundColor={[0.9, 0.9, 0.9, 1.0]} canvas={canvas}>
        <Camera>
          <Pass>
            <Plot>
              <Transform position={[200, 200]} scale={[200, -200]}>
                <Grid
                  first={{ divide: 20 }}
                  second={{ divide: 20 }}
                  width={2}
                  axes="xy"
                  color="#eee"
                ></Grid>
                <Axis axis="x" origin={[0, 0]} color="#fff" width={2}></Axis>
                <Axis axis="y" origin={[0, 0]} color="#fff" width={2}></Axis>
                <Point positions={points} colors={colors} size={10}></Point>
                <Line
                  zIndex={1}
                  width={2}
                  positions={pts}
                  color={[0.0, 0.0, 0.0, 0.6]}
                ></Line>
                {visualizeInlierSigma ? (
                  <InlierSigma
                    positions={points}
                    sigmas={inferenceResult.selectedModels.map(
                      (m) => m.inlier_sigma,
                    )}
                  />
                ) : null}
              </Transform>
            </Plot>
          </Pass>
        </Camera>
      </AutoCanvas>
    </WebGPU>
  )
}

type InlierSigmaProps = {
  positions: number[][]
  sigmas: number[]
}

const InlierSigma: LC<InlierSigmaProps> = (props: InlierSigmaProps) => {
  // For each position in positions, draws circles of each radius sigma centered at that point.
  const { positions, sigmas } = props
  const N = 45 // number of polygon segments to approximate circle
  const da = (2 * Math.PI) / N
  const pts = new Array(positions.length)
  for (let i = 0; i < positions.length; ++i) {
    pts[i] = new Array(sigmas.length)
    for (let j = 0; j < sigmas.length; ++j) {
      pts[i][j] = new Array(N)
    }
  }
  // Since we're drawing a repeated family of concentric
  // circles we do this inside out, generating the points
  // on the unit circle in the outer loop, then scaling them
  // by the sigma values, then translating them for each
  // point.
  for (let k = 0, a = 0; k < N; ++k, a += da) {
    const x = Math.cos(a)
    const y = Math.sin(a)
    for (let j = 0; j < sigmas.length; ++j) {
      const rx = sigmas[j] * x
      const ry = sigmas[j] * y
      for (let i = 0; i < positions.length; ++i) {
        pts[i][j][k] = [rx + positions[i][0], ry + positions[i][1]]
      }
    }
  }
  return (
    <Polygon positions={pts} width={2} stroke={[0.0, 1.0, 0.0, 1.0]}></Polygon>
  )
}

// Wrap this in its own component to avoid JSX trashing of the view
type CameraProps = PropsWithChildren<object>
const Camera: LC<CameraProps> = (props: CameraProps) => (
  /* 2D pan controls + flat view */
  <PanControls>
    {(x, y, zoom) => (
      <FlatCamera x={x} y={y} zoom={zoom}>
        {props.children}
      </FlatCamera>
    )}
  </PanControls>
)
