import globals from "globals"
import pluginJs from "@eslint/js"
import tseslint from "typescript-eslint"

export default [
  {
    ignores: ["eslint.config.mjs", ".venv/"],
  },
  {
    languageOptions: { globals: globals.browser },
    files: ["src/**/*.ts"],
  },
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,
]
