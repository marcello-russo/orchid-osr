export type SortDir = "asc" | "desc";
export type SortablePrimitive = string | number | boolean | null | undefined;

export function compareNullSafe<T extends SortablePrimitive>(
  a: T,
  b: T,
  dir: SortDir = "asc",
): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  if (a < b) return dir === "asc" ? -1 : 1;
  if (a > b) return dir === "asc" ? 1 : -1;
  return 0;
}
