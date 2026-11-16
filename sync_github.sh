#!/usr/bin/env bash
# ==============================================================================
# Automated Git & GitHub Synchronization Pipeline
# Tracks source scripts, documentation, and model checkpoint archives (.zip).
# ==============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=================================================================="
echo " [GitHub Sync] Starting Automated Commit & Push Pipeline"
echo " Target Directory: $REPO_DIR"
echo "=================================================================="

# Ensure Git repository exists
if [ ! -d ".git" ]; then
    echo "[*] Initializing git repository..."
    git init
    git branch -M main
fi

# Set default identity if unset
if [ -z "$(git config user.name || true)" ]; then
    git config user.name "Alimov Muxamad Sodiq"
fi
if [ -z "$(git config user.email || true)" ]; then
    git config user.email "r00tw4v3@r00tw4v3.r00tw4v3"
fi

# Enable credential storage so authentication persists
git config credential.helper store

# Verify remote configuration
REMOTE_URL=$(git remote get-url origin 2>/dev/null || true)
if [ -z "$REMOTE_URL" ]; then
    echo "[*] Adding remote origin https://github.com/r00tw4v3/rootpath-rl.git"
    git remote add origin https://github.com/r00tw4v3/rootpath-rl.git
fi

# If GITHUB_TOKEN is set in environment, inject it into remote URL
if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "[*] GITHUB_TOKEN detected; updating remote URL with token authentication..."
    git remote set-url origin "https://${GITHUB_TOKEN}@github.com/r00tw4v3/rootpath-rl.git"
fi

echo "[*] Staging core files, benchmarks, and model checkpoints..."
# Explicitly stage code, docs, and trained weights
git add .gitignore
git add -- *.py
git add -- *.sh
git add -- *.md
git add -f -- *.zip 2>/dev/null || true

# Extract winning metrics from BENCHMARK.md if available
METRICS_INFO="v3 DAG env + MaskablePPO best model"
if [ -f "BENCHMARK.md" ]; then
    BEST_LINE=$(grep -E "Success Rate:.*Mean Episode Length:" BENCHMARK.md 2>/dev/null || true)
    if [ -z "$BEST_LINE" ]; then
        SUCCESS=$(grep -m1 "Success Rate:" BENCHMARK.md | awk -F'`' '{print $2}' || true)
        EPLEN=$(grep -m1 "Mean Episode Length:" BENCHMARK.md | awk -F'`' '{print $2}' || true)
        if [ -n "$SUCCESS" ] && [ -n "$EPLEN" ]; then
            METRICS_INFO="v3 DAG env + MaskablePPO best model (ep_len=${EPLEN}, success=${SUCCESS})"
        fi
    fi
fi

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
COMMIT_MSG="feat(rl): ${METRICS_INFO} [${TIMESTAMP}]"

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "[*] No changes detected in working tree; working directory clean."
else
    echo "[*] Committing: ${COMMIT_MSG}"
    git commit -m "${COMMIT_MSG}"
fi

echo "[*] Pushing to GitHub (origin main)..."
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Check if non-interactive environment without credentials
if [ -z "${GITHUB_TOKEN:-}" ] && [ ! -t 0 ] && [ ! -f "$HOME/.git-credentials" ]; then
    echo "[!] Non-interactive session detected and no cached credentials found in ~/.git-credentials or GITHUB_TOKEN."
    echo "    Commit has been safely recorded locally in branch '${CURRENT_BRANCH}'."
    echo "    To push to GitHub, run:"
    echo "      export GITHUB_TOKEN=\"your_personal_access_token\""
    echo "      ./sync_github.sh"
    echo "    Or run directly from your terminal:"
    echo "      git push -u origin ${CURRENT_BRANCH}"
    exit 0
fi

if git push -u origin "$CURRENT_BRANCH"; then
    echo "=================================================================="
    echo " [GitHub Sync] Push completed successfully to origin/$CURRENT_BRANCH!"
    echo "=================================================================="
else
    echo "[!] Push encountered authentication or network issue." >&2
    echo "    To push manually: git push -u origin $CURRENT_BRANCH" >&2
    exit 1
fi
