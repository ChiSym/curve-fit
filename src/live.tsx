/* eslint-disable @typescript-eslint/no-unused-vars */

import type { LC, PropsWithChildren } from "@use-gpu/live"
import { into } from "@use-gpu/live"
import { AutoCanvas, WebGPU } from "@use-gpu/webgpu"
//import { Cartesian, Axis, Grid } from "@use-gpu/plot"
import { HTML } from "@use-gpu/react"
import { Animate, Environment, FlatCamera, FontLoader, LinearRGB, LineLayer, PanControls, Pass, PointLayer } from "@use-gpu/workbench"
import { Block, Flex, Inline, Layout, UI, Text } from "@use-gpu/layout"
import { FALLBACK_MESSAGE } from "./fallback"
import { Axis, Cartesian, Grid, Label, Line, Plot, Point, Sampler, Scale, Scissor, Surface, Tick, Transform, Transpose } from "@use-gpu/plot"
import { InferenceResult } from "./gpgpu"


type ComponentProps = {
  canvas: HTMLCanvasElement
  inferenceResult: InferenceResult
}

// const FONTS = [
//   {
//     family: 'Lato',
//     weight: 'black',
//     style: 'normal',
//     src: '/Lato-Black.ttf',
//   },
// ];

const xs = new Array(60)
const h = 2.0/(xs.length-1)
for (let i = 0, x = -1; i < xs.length; ++i, x += h) xs[i] = x


export const Component: LC<ComponentProps> = (
  props: PropsWithChildren<ComponentProps>,
) => {
  const { canvas, inferenceResult } = props
  function modelToFn(model: Float32Array) {
    return (x: number) => {
      const [a0, a1, a2, omega, A, phi] = model
      return a0 + x*a1 + x*x*a2 + A * Math.sin(phi + omega * x)
    }
  }
  const pts: number[][][] = inferenceResult.selectedModels.map(m => {
    const f = modelToFn(m.model)
    return xs.map(x => [x, f(x)])
  })
  return (
    <WebGPU
      fallback={(error: Error) => <HTML>{into(FALLBACK_MESSAGE(error))}</HTML>}
    >
      <FontLoader>
        <AutoCanvas canvas={canvas}>
          <Camera>
            <Pass>
              <UI>
                <Layout>
                  <Flex height={400} width={400} align="between" fill="#ff000020">
                    <Flex width={300} height={180} fill="#00ff0030" direction="y">
                      <Inline>
                        <Text
                          size={48}
                          detail={64}
                          snap={false}
                          text={"Hello World"}
                          family="Lato"
                          color="#ff00ff"
                        />
                      </Inline>
                      <Flex width={300} height={100} fill="#0000ff30">
                      </Flex>
                    </Flex>
                  </Flex>
                </Layout>
              </UI>
              <Plot>
                <Grid axes="xy">

                </Grid>
                <Transform position={[200,200]} scale={[200, -200]}>
                  <Point position={[-0.9,-0.9]} shape='circle' size={10} color={'#f00'}></Point>
                  <Point position={[0.9,-0.9]} shape='circle' size={10} color={'#0f0'}></Point>
                  <Point position={[-0.9,0.9]} shape='circle' size={10} color={'#00f'}></Point>
                  <Point position={[0.9,0.9]} shape='circle' size={10} color={'#f0f'}></Point>
                  <Line positions={pts} color={'#ff0'}></Line>
                  <Line positions={[[-0.9,-0.9], [0.9,0.9]]} color={'#fff'}></Line>
                </Transform>
              </Plot>
              <PointLayer shape='circle' position={[0,0]} size={10} color={[0.8,0.8,1.0,1.0]}>
              </PointLayer>
            </Pass>
          </Camera>
        </AutoCanvas>
      </FontLoader>
    </WebGPU>
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
