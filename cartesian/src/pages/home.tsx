import type { LiveComponent } from "@use-gpu/live"
import { FC, PropsWithChildren } from "react"

import { use } from "@use-gpu/live"
import { HTML } from "@use-gpu/react"
import { useRouterContext } from "@use-gpu/workbench"

import React from "react"
import { styled } from "@stitches/react"

import { makePages } from "../routes"

export const Title: FC<any> = styled("h1", {
  margin: 0,
  padding: "10px 20px",
  textAlign: "center",
  fontSize: "24px",
  lineHeight: "48px",
} as any)

export type HomePageProps = {
  container?: Element | null
}

const PANEL_STYLE = {
  position: "absolute",
  left: "50%",
  top: "50%",
  width: "400px",
  height: "600px",
  marginLeft: "-200px",
  marginTop: "-300px",
  background: "rgba(0, 0, 0, .75)",
  zIndex: 1,
}

export const HomePage: LiveComponent<HomePageProps> = (props) => {
  const { container } = props
  const { linkTo, replace } = useRouterContext()

  const PAGES = makePages()

  return use(HTML, {
    container,
    style: PANEL_STYLE,
    children: (
      <>
        <Title>Use.GPU Demos</Title>

        <ul>
          {PAGES.slice(0, -1).map(({ title, path }) => (
            <li key={path}>
              <a {...linkTo(path)}>{title}</a>
            </li>
          ))}
        </ul>
      </>
    ),
  })
}
