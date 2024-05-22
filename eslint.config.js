import globals from "globals"
import pluginJs from "@eslint/js"
import tseslint from "typescript-eslint"

export default [
  {
    languageOptions: { globals: globals.browser },
    files: ["src/**/*.ts"],
    ignores: ["eslint.config.js", ".venv/"],
  },
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,
]
