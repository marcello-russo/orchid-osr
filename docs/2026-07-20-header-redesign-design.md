# Header Redesign — Design Doc

## Objective

Simplify the layout header to be minimal (centered title only) and move all auxiliary controls (user info, language toggle, theme toggle) into the sidebar footer.

## Motivation

The current header is crowded: hamburger menu, logo, username/logout, language toggle, and theme toggle all compete for space in a 52px bar. This makes the header feel busy and reduces the visual clarity of the app.

## Changes

### Header (`Header.tsx`)

- **Before:** Left=hamburger, Center=logo, Right=username+logout+LangToggle+ThemeToggle
- **After:** Left=hamburger, Center=logo, Right=(empty)
- The hamburger remains the only interactive element in the header (opens the sidebar drawer)
- The logo text is centered exactly as before

### Sidebar (`Sidebar.tsx`)

- **New section at bottom:** A `sidebar-footer` area below the navigation items, separated by a subtle divider
- **User row:** Avatar circle + username on the left, "Log out" button on the right
- **Actions row:** Language toggle button (EN/IT) and Theme toggle button (Light/Dark) side by side
- The footer sits below all nav items, pinned to the bottom of the sidebar drawer

### Layout (`Layout.tsx`)

- No structural changes needed — the sidebar already renders inside a flex column that can accommodate a footer

### State handling

- If the user is not authenticated: the sidebar footer shows "Log in" instead of the user row
- This replaces the current log-in link in the header

## Files to modify

| File | Change |
|------|--------|
| `src/components/layout/Header.tsx` | Remove right section (user, lang, theme); keep only hamburger + logo |
| `src/components/layout/Sidebar.tsx` | Add footer section with user row + actions row |
| `src/components/layout/LanguageToggle.tsx` | No change (reused in sidebar) |
| `src/components/layout/ThemeToggle.tsx` | No change (reused in sidebar) |

## Testing

- Visual: open the app, verify header is clean with only hamburger + centered logo
- Visual: open the sidebar, verify footer shows user info + lang toggle + theme toggle
- Functional: click theme toggle in sidebar — theme switches correctly
- Functional: click lang toggle in sidebar — language switches correctly
- Functional: click "Log out" — session ends, sidebar footer shows "Log in"
- Functional: click "Log in" — navigates to `/auth`
- Mobile: sidebar drawer opens/closes correctly with footer visible
