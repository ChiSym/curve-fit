import type { LC, PropsWithChildren } from "@use-gpu/live"

import React, { use } from "@use-gpu/live"
import { TensorArray } from "@use-gpu/core"

import {
  Loop,
  Pass,
  OrbitCamera,
  OrbitControls,
  Cursor,
  Environment,
  Animate,
  LinearRGB,
} from "@use-gpu/workbench"
import {
  Plot,
  Cartesian,
  Axis,
  Grid,
  Label,
  Line,
  Sampler,
  Scale,
  Scissor,
  Surface,
  Tick,
  Transpose,
} from "@use-gpu/plot"
import { vec3 } from "gl-matrix"

let t = 0

const BACKGROUND = [0, 0, 0.09, 1]

export const PlotCartesianPage: LC = () => {
  return (
    <>
      <LinearRGB backgroundColor={BACKGROUND} tonemap="aces" gain={2}>
        <Cursor cursor="move" />
        <Camera>
          <Pass>
            <Environment preset="park">
              <Plot>
                <Animate
                  loop
                  mirror
                  delay={0}
                  keyframes={[
                    [
                      0,
                      [
                        [-3, 0],
                        [0, 1],
                        [0, 3],
                      ],
                    ],
                    [
                      10,
                      [
                        [0, 3],
                        [0, 1],
                        [0, 3],
                      ],
                    ],
                  ]}
                  prop="range"
                >
                  <Cartesian scale={[2, 1, 1]}>
                    <Grid
                      axes="xy"
                      width={2}
                      first={{ detail: 3, divide: 5 }}
                      second={{ detail: 3, divide: 5 }}
                      depth={0.5}
                      zBias={-1}
                      auto
                    />
                    <Grid
                      axes="xz"
                      width={2}
                      first={{ detail: 3, divide: 5 }}
                      second={{ detail: 3, divide: 5 }}
                      depth={0.5}
                      zBias={-1}
                    />
                    <Grid
                      axes="yz"
                      width={2}
                      first={{ detail: 3, divide: 5 }}
                      second={{ detail: 3, divide: 5, end: true }}
                      depth={0.5}
                      zBias={-1}
                      auto
                    />

                    <Axis
                      axis="x"
                      width={5}
                      color={[0.75, 0.75, 0.75, 1]}
                      depth={0.5}
                      end
                    />
                    <Scale origin={[0, 0, 3]} divide={5} axis="x">
                      <Tick
                        size={10}
                        width={2}
                        offset={[0, 1, 0]}
                        color={[0.75, 0.75, 0.75, 1]}
                        depth={0.5}
                      />
                      <Label
                        placement="bottom"
                        color="#40406080"
                        size={24}
                        offset={16}
                        expand={5}
                        depth={0.5}
                      />
                      <Label
                        placement="bottom"
                        color="#ffffff"
                        size={24}
                        offset={16}
                        expand={0}
                        depth={0.5}
                      />
                    </Scale>

                    <Axis
                      axis="y"
                      width={5}
                      color={[0.75, 0.75, 0.75, 1]}
                      detail={8}
                      depth={0.5}
                      end
                    />
                    <Axis
                      axis="z"
                      width={5}
                      color={[0.75, 0.75, 0.75, 1]}
                      detail={8}
                      depth={0.5}
                      end
                    />

                    <Scissor>
                      <Sampler
                        axes="zx"
                        format="vec4<f32>"
                        size={[10, 20]}
                        origin={[0, 0, 0]}
                        items={2}
                        as={["positions", "colors"]}
                        expr={(emit, z, x) => {
                          const v = Math.cos(x) * Math.cos(z)
                          emit(x, v * 0.4 + 0.5, z, 1)

                          const r = Math.max(0.0, v * v * v)
                          const g = Math.max(0.0, -v) * v
                          const b = 0.25 + 0.75 * Math.abs(v)
                          emit(r, g, b, 1)
                        }}
                      >
                        {({
                          positions,
                          colors,
                        }: Record<string, TensorArray>) => (
                          <>
                            <Surface positions={positions} colors={colors} />
                            <Line
                              positions={positions}
                              width={2}
                              color={[0.5, 0.5, 1, 0.25]}
                              depth={0.5}
                              zBias={1}
                              blend="add"
                            />
                            <Transpose tensor={positions} axes="yx">
                              <Line
                                width={2}
                                color={[0.5, 0.5, 1, 0.25]}
                                depth={0.5}
                                zBias={1}
                                blend="add"
                              />
                            </Transpose>
                          </>
                        )}
                      </Sampler>
                    </Scissor>
                  </Cartesian>
                </Animate>
              </Plot>
            </Environment>
          </Pass>
        </Camera>
      </LinearRGB>
    </>
  )
}

const Camera = ({ children }: PropsWithChildren<object>) => (
  <OrbitControls
    radius={5}
    bearing={0.5}
    pitch={0.3}
    render={(radius: number, phi: number, theta: number, target: vec3) => (
      <OrbitCamera radius={radius} phi={phi} theta={theta} target={target}>
        {children}
      </OrbitCamera>
    )}
  />
)
