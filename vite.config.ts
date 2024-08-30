import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import wgsl from '@use-gpu/wgsl-loader/rollup'


// function myWgsl(options?) {
//   const r = wgsl(options)
//   return {
//     name: r.name,
//     transform(src: string, id: string) {
//       const t = r.transform(src, id)
//       if (!t) return t
//       return {
//         code: t.code,
//         map: null
//       }
//     }
//   }
// }

//import { createFilter } from 'rollup-pluginutils'
//import { transpileWGSL } from '@use-gpu/shader/wgsl'
//import MagicString from 'magic-string'


// const myWgslLoader = (userOptions?) => {
//   const options = Object.assign({
//     exclude: [],
//     include: ['**/*.wgsl']
//   }, userOptions);
//   const filter = createFilter(options)
//   return {
//     name: '@use-gpu/wgsl-loader',
//     transform: (src: string, id: string) => {
//       if (!filter(id)) return
//       const code = transpileWGSL(src, id, true)
//       const magicString = new MagicString(code)
//       return {
//         code: magicString.toString(),
//         map: null
//       }
//     }
//   }
// }


export default defineConfig({
  plugins: [react(), wgsl()],
})
