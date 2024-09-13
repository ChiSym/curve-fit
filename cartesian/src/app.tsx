import type { LC } from "@use-gpu/live"

import React, {
  hot,
  into,
  useFiber,
  useMemo,
  useOne,
  useResource,
  useState,
} from "@use-gpu/live"
import { HTML } from "@use-gpu/react"
import { AutoCanvas, FPSCounter, WebGPU } from "@use-gpu/webgpu"
import {
  DebugProvider,
  FontLoader,
  Router,
  Routes,
  useKeyboard,
} from "@use-gpu/workbench"

import { UseInspect } from "@use-gpu/inspect"
import { inspectGPU } from "@use-gpu/inspect-gpu"
import "@use-gpu/inspect/theme.css"

import { makeRoutes } from "./routes"

import { FALLBACK_MESSAGE } from "./fallback"

import NOTO_SEQUENCES from "./noto-emoji"

// @ts-ignore
const isDevelopment = process.env.NODE_ENV === "development"

const base = isDevelopment ? "/" : "/demo/"

// Toggle inspector with ctrl/cmd-I.
// Trigger re-render with ctrl/cmd-J.
const useInspector = () => {
  const [version, setVersion] = useState<number>(0)
  const [inspect, setInspect] = useState<boolean>(true)

  useResource((dispose) => {
    const keydown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "i") setInspect((s) => !s)
      if ((e.ctrlKey || e.metaKey) && e.key === "j") setVersion((s) => s + 1)
    }

    window.addEventListener("keydown", keydown)
    dispose(() => window.addEventListener("keydown", keydown))
  })

  return inspect
}

export const FPSToggle = () => {
  const [fps, setFPS] = useState(false)
  const { keyboard } = useKeyboard()
  useOne(() => keyboard.keys.f && setFPS(!fps), keyboard.keys.f)
  return fps ? <FPSCounter container="#use-gpu > .canvas" top={32} /> : null
}

export const App: LC = () => {
  const root = document.querySelector("#use-gpu")!
  const inner = document.querySelector("#use-gpu .canvas")!

  const router = useOne(
    () => (
      <Router base={base}>
        <Routes routes={makeRoutes()} morph />
      </Router>
    ),
    root,
  )

  const fonts = useOne(() => [
    {
      family: "Lato",
      weight: 400,
      style: "normal",
      src: base + "fonts/Lato-Regular.ttf",
    },
    {
      family: "Lato",
      weight: 400,
      style: "italic",
      src: base + "fonts/Lato-Italic.ttf",
    },
    {
      family: "Lato",
      weight: 500,
      style: "normal",
      src: base + "fonts/Lato-Bold.ttf",
    },
  ])

  const fiber = useFiber()
  const inspect = useInspector()

  const view = useMemo(
    () => (
      <WebGPU
        fallback={(error: Error) => (
          <HTML container={inner}>{into(FALLBACK_MESSAGE(error))}</HTML>
        )}
      >
        <AutoCanvas autofocus selector={"#use-gpu .canvas"} samples={4}>
          <FontLoader fonts={fonts}>{router}</FontLoader>
          <FPSToggle />
        </AutoCanvas>
      </WebGPU>
    ),
    [root, fonts, router],
  )

  return (
    <UseInspect
      fiber={fiber}
      container={root}
      active={inspect}
      provider={DebugProvider}
      extensions={[inspectGPU]}
    >
      {view}
    </UseInspect>
  )
  // @ts-ignore
}

App.displayName = "App"
