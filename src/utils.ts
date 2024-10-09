// Utility type for our object maps
export type TypedObject<T> = { [key: string]: T }

// Utility functions
export function setItem<T>(
  obj: TypedObject<T>,
  key: string,
  value: T,
): TypedObject<T> {
  return { ...obj, [key]: value }
}
