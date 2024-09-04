import React from "@use-gpu/live"

import { PlotCartesianPage } from "./pages/plot/cartesian"

import { HomePage } from "./pages/home"
import { EmptyPage } from "./pages/empty"

export const makePages = () => [
  {
    path: "/plot/cartesian",
    title: "Plot - XYZ",
    element: <PlotCartesianPage />,
  },

  {
    path: "/",
    title: "Index",
    element: <HomePage container={document.querySelector("#use-gpu")} />,
  },
]

export const makeRoutes = () => ({
  ...makePages().reduce(
    (out, { path, element }) => ((out[path] = { element }), out),
    {} as Record<string, any>,
  ),
  "*": { element: <EmptyPage /> },
})
