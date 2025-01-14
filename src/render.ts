import * as THREE from "three"
import { InferenceResult } from "./gpgpu"

export class Render {
  private camera: THREE.Camera
  private renderer: THREE.WebGLRenderer
  inlier_color = new THREE.Color(0x00ff00)
  outlier_color = new THREE.Color(0xff0000)

  constructor(canvas: HTMLCanvasElement) {
    this.camera = new THREE.OrthographicCamera()
    this.camera.position.set(0, 0, 1)
    this.camera.lookAt(0, 0, 0)
    this.renderer = new THREE.WebGLRenderer({ canvas: canvas })
    const { width, height } = canvas.getBoundingClientRect()
    this.renderer.setSize(width, height)
    this.renderer.setClearColor(0xffffff)
  }

  private modelToFn(model: Float32Array) {
    return (x: number) => {
      const [a0, a1, a2, omega, A, phi] = model
      return a0 + x * a1 + x * x * a2 + A * Math.sin(phi + omega * x)
    }
  }

  render(
    inferenceResult: InferenceResult,
    points: number[][],
    vizInlierSigma: boolean,
  ) {
    const mat = new THREE.LineBasicMaterial({ color: 0x333 })
    const inlierMat = new THREE.LineBasicMaterial({ color: 0x00ff00 })
    const scene = new THREE.Scene()
    const outlierCounts = new Array(points.length).fill(0)
    inferenceResult.selectedModels.forEach((m) => {
      const f = this.modelToFn(m.model)
      const N = 50
      const pts = Array(N)
      for (let i = 0; i < N; ++i) {
        const x = i / 25 - 1
        const y = f(x)
        pts[i] = new THREE.Vector3(x, y, 0)
      }
      scene.add(
        new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat),
      )
      for (let b = 1, i = 0; i < points.length; ++i, b <<= 1) {
        if ((m.outlier & b) != 0) {
          ++outlierCounts[i]
        }
      }
      if (vizInlierSigma) {
        points.forEach((p) => {
          const circ = new THREE.EllipseCurve(
            p[0],
            p[1],
            m.inlier_sigma,
            m.inlier_sigma,
          )
          const pts = circ.getSpacedPoints(24)
          scene.add(
            new THREE.LineLoop(
              new THREE.BufferGeometry().setFromPoints(pts),
              inlierMat,
            ),
          )
        })
      }
    })
    points.forEach((p, i) => {
      const circ = new THREE.CircleGeometry(0.03, 10)
      circ.translate(p[0], p[1], 0)
      const mat_c = new THREE.LineBasicMaterial({
        color: new THREE.Color().lerpColors(
          this.inlier_color,
          this.outlier_color,
          outlierCounts[i] / inferenceResult.selectedModels.length,
        ),
      })
      const circ_mesh = new THREE.Mesh(circ, mat_c)
      scene.add(circ_mesh)
    })

    this.renderer.render(scene, this.camera)
  }
}
