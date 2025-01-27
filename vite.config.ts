import { defineConfig } from "vite"
import checker from "vite-plugin-checker"
import eslint from "vite-plugin-eslint"
import react from "@vitejs/plugin-react"
import wasm from "vite-plugin-wasm"
import { resolve } from "path"
import virtual from "vite-plugin-virtual"
import topLevelAwait from "vite-plugin-top-level-await"
import { execSync } from "child_process"

export default defineConfig({
  base: "",
  plugins: [
    wasm(),
    virtual({
      'virtual:version': `export default '${execSync('git describe --tags --always').toString().replace(/\s/, '')}'`
    }),
    topLevelAwait(),
    react(),
    eslint(),
    checker({
      typescript: true,
    }),
  ],
  server: {
    port: 8080,
  },
  preview: {
    port: 8080,
  },
  build: {
    outDir: "./dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        stat: resolve(__dirname, 'stat/index.html'),
        main: resolve(__dirname, './index.html')
      }
    }
  },
})
