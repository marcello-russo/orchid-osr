import { useQuery } from "@tanstack/react-query";
import { fetchSpecies } from "@/lib/api";

export function useSpecies() {
  return useQuery({
    queryKey: ["species"],
    queryFn: fetchSpecies,
  });
}
