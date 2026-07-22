import { createContext, useContext, useState, type ReactNode } from "react";

interface CiteContextValue {
  activeCite: string | null;
  setActiveCite: (id: string | null) => void;
}

const CiteContext = createContext<CiteContextValue | null>(null);

export function CiteProvider({ children }: { children: ReactNode }) {
  const [activeCite, setActiveCite] = useState<string | null>(null);
  return (
    <CiteContext.Provider value={{ activeCite, setActiveCite }}>
      {children}
    </CiteContext.Provider>
  );
}

export function useCiteContext(): CiteContextValue {
  const ctx = useContext(CiteContext);
  if (!ctx) throw new Error("useCiteContext must be used inside CiteProvider");
  return ctx;
}
