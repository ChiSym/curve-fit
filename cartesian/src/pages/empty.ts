import type { LiveComponent } from "@use-gpu/live"
import { use } from "@use-gpu/live"

import { Pass } from "@use-gpu/workbench"

export type EmptyPageProps = {
  _unused?: boolean
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export const EmptyPage: LiveComponent<EmptyPageProps> = (_: EmptyPageProps) => {
  return use(Pass, {})
}
