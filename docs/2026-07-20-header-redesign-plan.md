# Header Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the header to show only centered logo + hamburger, move user/lang/theme controls to sidebar footer.

**Architecture:** Two-component change: `Header.tsx` removes right section, `Sidebar.tsx` adds a footer below nav items with user row + actions row.

**Tech Stack:** React + TypeScript + Tailwind + Zustand + lucide-react

---

### Task 1: Header — remove right section

**Files:**
- Modify: `frontend/src/components/layout/Header.tsx`

- [ ] **Step 1: Read current Header.tsx**

Read the current file to see the right section with user info, LanguageToggle, ThemeToggle.

- [ ] **Step 2: Remove right section**

Remove everything after the center logo span. The header should contain only:
- Hamburger button (left)
- Logo "OrchidOSR NCD" (center)
- Empty div as spacer (right) to keep centering

The key change: delete the `{token ? (...)} : (...)}` block and the `{t("header.logout")}` button, and the LanguageToggle/ThemeToggle imports.

Import `Menu` from `lucide-react` (already likely there). The final header JSX structure:

```tsx
<header className="fixed top-0 right-0 left-0 z-30 h-14 bg-card/80 backdrop-blur-md border-b">
  <div className="flex justify-between items-center h-full px-4">
    <button
      onClick={toggleSidebar}
      aria-expanded={sidebarOpen}
      aria-controls="sidebar"
      className="p-2 hover:bg-muted rounded-md transition-colors"
    >
      <Menu className="h-5 w-5" />
    </button>

    <div className="flex items-center gap-1.5">
      <img src={favicon} alt="" className="h-5 w-5" />
      <span className="text-primary font-bold text-lg">Orch</span>
      <span className="text-foreground font-bold text-lg">ID</span>
      <span className="text-muted-foreground font-bold text-xs ml-0.5">NCD</span>
    </div>

    <div className="w-10" />
  </div>
</header>
```

Remove unused imports: `useAppStore` (for token/user), `LanguageToggle`, `ThemeToggle`, `t` from `useTranslation` (if no longer needed), `LogOut`, `User` icons.

- [ ] **Step 3: Clean up imports**

```tsx
import { useSidebarStore } from "@/stores/useAppStore";
import { Menu } from "lucide-react";
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/layout/Header.tsx
git commit -m "refactor: header solo hamburger + logo centrato"
```

---

### Task 2: Sidebar — add footer with user + actions

**Files:**
- Modify: `frontend/src/components/layout/Sidebar.tsx`

- [ ] **Step 1: Read current Sidebar.tsx**

Read the current file to understand imports and existing structure (nav items after logo header).

- [ ] **Step 2: Add footer section after nav**

After the closing `</nav>` tag, add a sidebar footer with:
- Divider line
- User row (if authenticated): avatar + username + logout button
- Actions row: LanguageToggle + ThemeToggle side by side

```tsx
{/* Footer */}
<div className="border-t mt-auto">
  <div className="p-3 space-y-2">
    {user ? (
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-full bg-primary/20 flex items-center justify-center text-sm font-semibold text-primary">
            {user.username.charAt(0).toUpperCase()}
          </div>
          <span className="text-sm font-medium">{user.username}</span>
        </div>
        <button
          onClick={logout}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {t("header.logout")}
        </button>
      </div>
    ) : (
      <Link
        to="/auth"
        onClick={() => setSidebarOpen(false)}
        className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <LogIn className="h-4 w-4" />
        <span>{t("header.login")}</span>
      </Link>
    )}

    <div className="flex gap-2">
      <LanguageToggle />
      <ThemeToggle />
    </div>
  </div>
</div>
```

The sidebar's container should use `flex flex-col h-full` so the footer is pushed to the bottom via `mt-auto`.

- [ ] **Step 3: Add imports**

```tsx
import { Link } from "react-router-dom";
import { LogIn } from "lucide-react";
import LanguageToggle from "./LanguageToggle";
import ThemeToggle from "./ThemeToggle";
```

Import `useAppStore` for `user`, `logout`, `sidebarOpen`, `setSidebarOpen`.

- [ ] **Step 4: Ensure sidebar container is flex column with full height**

The sidebar `aside` element should have `flex flex-col h-full` so `mt-auto` on the footer pushes it down.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/layout/Sidebar.tsx
git commit -m "feat: sidebar footer con user, lang, theme"
```

---

### Task 3: Verify build

- [ ] **Step 1: Build frontend**

```bash
cd frontend && npm run build
```

Expected: no TypeScript errors, build succeeds.

- [ ] **Step 2: Review visually**

Open the app, verify:
- Header has only hamburger + centered logo
- Sidebar footer shows user info + logout + lang/theme toggles
- Logged-out state shows "Log in" link
- All toggles still work
