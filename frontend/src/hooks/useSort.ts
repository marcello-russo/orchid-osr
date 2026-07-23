import { useReducer, useMemo, useCallback } from "react";
import { compareNullSafe, type SortDir, type SortablePrimitive } from "@/lib/sort";

function getSortableValue<T, K extends keyof T>(obj: T, key: K): SortablePrimitive {
  return obj[key] as SortablePrimitive;
}

interface SortState<K> {
  key: K;
  dir: SortDir;
}

type SortAction<K> = { key: K };

function sortReducer<K>(state: SortState<K>, action: SortAction<K>): SortState<K> {
  if (action.key === state.key) {
    return { key: action.key, dir: state.dir === "asc" ? "desc" : "asc" };
  }
  return { key: action.key, dir: "asc" };
}

export function useSort<T, K extends keyof T>(items: T[], defaultKey: K) {
  const [state, dispatch] = useReducer(sortReducer<K>, {
    key: defaultKey,
    dir: "asc",
  });

  const sorted = useMemo(() => {
    const copy = [...items];
    copy.sort((a, b) => compareNullSafe(
      getSortableValue(a, state.key),
      getSortableValue(b, state.key),
      state.dir,
    ));
    return copy;
  }, [items, state.key, state.dir]);

  const toggle = useCallback((key: string) => dispatch({ key: key as unknown as K }), []);

  return {
    key: state.key as K,
    dir: state.dir,
    toggle,
    sorted,
  };
}
