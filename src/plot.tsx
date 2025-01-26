import "@react-three/fiber"
import { useFrame, useThree } from "@react-three/fiber"
import { RefObject, useEffect, useMemo, useState } from "react"
import { Color, OrthographicCamera, Vector3 } from "three"
import { Circle, Instance, Instances, Line } from "@react-three/drei"
import { Model } from "./model"
import { Animator } from "./animator"

export function Plot({
  animatorRef,
  points,
  vizInlierSigma,
}: {
  animatorRef: RefObject<Animator>
  points: number[][]
  vizInlierSigma: boolean
}) {
  const { camera } = useThree()

  const [models, setModels] = useState<Model[]>([])

  useEffect(() => {
    console.log("setting up camera", camera)
    const o = camera as OrthographicCamera
    o.position.set(0, 0, 5)
    o.left = -1
    o.right = 1
    o.top = 1
    o.bottom = -1
    camera.lookAt(new Vector3(0, 0, 0))
    camera.updateProjectionMatrix()
  }, [])

  useFrame(() => {
    if (animatorRef.current) {
      const r = animatorRef.current.awaitResult()
      setModels(r.inferenceResult.selectedModels)
    }
  })

  return (
    <>
      <Axes />
      <CurvePoints models={models} points={points}></CurvePoints>
      {vizInlierSigma && <InlierCircles models={models} points={points} />}
      <Curves models={models}></Curves>
    </>
  )
}

function Axes() {
  const vertices = useMemo(() => {
    const N = 10
    const vertices: [x: number, y: number][] = []
    for (let i = -N; i <= N; ++i) {
      for (let j = -N; j <= N; ++j) {
        const x = i / N
        const y = j / N
        vertices.push([x, -1])
        vertices.push([x, 1])
        vertices.push([-1, y])
        vertices.push([1, y])
      }
    }
    return vertices
  }, [])

  return <Line segments points={vertices}></Line>
}

function CurvePoints({
  models,
  points,
}: {
  models: Model[]
  points: number[][]
}) {
  const inlierCounts = new Array(points.length).fill(0)
  models.forEach((m) => {
    for (let z = 1, i = 0; i < points.length; ++i, z <<= 1) {
      // TODO: this nonsense is spread throughout the codebase. Make it a method of Model
      // so that we only have to do this once & the weird implementation detail of the
      // outlier bitfield doesn't have to be understood by everyone
      if (m.outlier & z) ++inlierCounts[i]
    }
  })
  const c0 = new Color(0, 0, 1)
  const c1 = new Color(1, 0, 0)
  const colors = inlierCounts.map((k) => {
    const c = new Color()
    c.lerpColors(c0, c1, k / points.length)
    return c
  })

  return points.map((p, i) => (
    <Circle position={[p[0], p[1], 0]} args={[0.03, 12]}>
      <meshBasicMaterial transparent={true} opacity={0.7} color={colors[i]} />
    </Circle>
  ))
}

function InlierCircles({
  models,
  points,
}: {
  models: Model[]
  points: number[][]
}) {
  const pts = useMemo(() => AnnularGeometry(24, 0.03), [])
  return (
    <Instances>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={pts}
          itemSize={3}
          count={pts.length / 3}
        ></bufferAttribute>
      </bufferGeometry>
      <lineBasicMaterial color="green"></lineBasicMaterial>
      {points.flatMap((p) =>
        models.map((m) => (
          <Instance
            position={[p[0], p[1], 0]}
            scale={m.inlier_sigma}
          ></Instance>
        )),
      )}
    </Instances>
  )
}

function Curves({ models }: { models: Model[] }) {
  function curve_points(m: Model): [x: number, y: number][] {
    const N = 15
    const pts = new Array(2 * N + 1)
    for (let i = -N, j = 0; i <= N; ++i, ++j) {
      pts[j] = [i / N, m.f(i / N)]
    }
    return pts
  }
  return models.map((m, i) => (
    <Line
      key={i}
      opacity={0.3}
      lineWidth={3}
      color="black"
      points={curve_points(m)}
    />
  ))
}

// R3F seems to be missing a way to produce an "instanced line".
// Instead we make a mesh for an annular region, so that we get
// and "instance-able circle" with variable thickness. The inner
// and outer circles are drawn with N points, and the annulus is
// bounded by the radii [1, 1+dr]
function AnnularGeometry(N: number, dr: number) {
  const innerCircle = Array(N + 1)
  const outerCircle = Array(N + 1)
  const drp1 = dr + 1
  for (let i = 0; i <= N; ++i) {
    const t = (2 * Math.PI * i) / N
    innerCircle[i] = [Math.cos(t), Math.sin(t)]
    outerCircle[i] = [drp1 * Math.cos(t), drp1 * Math.sin(t)]
  }
  const pts = new Float32Array(N * 6 * 3)
  let j = 0
  for (let i = 0; i < N; ++i) {
    // Small rectangular region of the annulus bounded by
    // r \in [1, 1+dr] and theta \in [t, t+dt], formed by
    // two triangles. Triangle 1:
    pts[j++] = innerCircle[i][0]
    pts[j++] = innerCircle[i][1]
    pts[j++] = 0
    pts[j++] = outerCircle[i][0]
    pts[j++] = outerCircle[i][1]
    pts[j++] = 0
    pts[j++] = outerCircle[i + 1][0]
    pts[j++] = outerCircle[i + 1][1]
    pts[j++] = 0
    // Triangle 2:
    pts[j++] = innerCircle[i][0]
    pts[j++] = innerCircle[i][1]
    pts[j++] = 0
    pts[j++] = outerCircle[i + 1][0]
    pts[j++] = outerCircle[i + 1][1]
    pts[j++] = 0
    pts[j++] = innerCircle[i + 1][0]
    pts[j++] = innerCircle[i + 1][1]
    pts[j++] = 0
  }
  return pts
}
