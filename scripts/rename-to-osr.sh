#!/usr/bin/env bash
# Rename orchid-ncd → orchid-osr in all source files
# NON tocca: HF repo IDs, Docker volume names, git history
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Renaming orchid-ncd → orchid-osr ==="

# 1. Text replacements in source files (NOT Docker volumes, NOT HF repos)
# Pattern: lowercase "orchid-ncd" in text
grep -rl "orchid-ncd" --include="*.py" --include="*.ts" --include="*.tsx" \
  --include="*.json" --include="*.md" --include="*.yml" --include="*.yaml" \
  --include="*.sh" --include="*.html" --include="*.svg" --include="*.conf" \
  --include="Dockerfile" --include="*.css" --include="*.env*" \
  2>/dev/null | grep -v "node_modules" | grep -v ".git/" | grep -v "__pycache__" \
  | grep -v "coverage/" | grep -v "package-lock" \
  | while read f; do
    # Skip HF repo URLs (they don't exist with new name)
    if grep -q "huggingface.co/marcellorusso/orchid-ncd" "$f" 2>/dev/null; then
      echo "  ⚠️  SKIP (HF repo): $f"
      continue
    fi
    # Skip Docker volume references
    if grep -q "orchid-ncd_orchid-data\|orchid-ncd_nginx\|project_name" "$f" 2>/dev/null; then
      echo "  ⚠️  SKIP (Docker volume): $f"
      continue
    fi
    sed -i 's/orchid-ncd/orchid-osr/g' "$f"
    echo "  ✅ $f"
  done

# 2. PascalCase "OrchID OSR" → "OrchID OSR" (JWT claims, display names)
grep -rl "OrchID OSR" --include="*.py" --include="*.ts" --include="*.tsx" \
  --include="*.json" --include="*.md" --include="*.yml" --include="*.yaml" \
  --include="*.sh" --include="*.html" 2>/dev/null | grep -v "node_modules" \
  | grep -v ".git/" | grep -v "__pycache__" | grep -v "coverage/" \
  | while read f; do
    sed -i 's/OrchID OSR/OrchID OSR/g' "$f"
    echo "  ✅ (OrchID OSR) $f"
  done

# 3. Uppercase "orchid_ncd" in Python  
grep -rl "orchid_ncd" --include="*.py" 2>/dev/null | grep -v "node_modules" \
  | grep -v ".git/" | grep -v "__pycache__" | while read f; do
  sed -i 's/orchid_ncd/orchid_osr/g' "$f"
  echo "  ✅ (underscore) $f"
done

echo ""
echo "=== Fatto! ==="
echo "Ora esegui: git diff --stat per verificare le modifiche"
