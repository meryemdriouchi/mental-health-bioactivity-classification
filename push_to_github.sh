#!/bin/bash
# Push mental-health-bioactivity-classification to GitHub
# Usage:
#   Option A (recommended): export GITHUB_TOKEN=your_token && ./push_to_github.sh
#   Option B: Create empty repo at https://github.com/new?name=mental-health-bioactivity-classification
#             then run: ./push_to_github.sh

set -e

REPO_OWNER="meryemdriouchi"
REPO_NAME="mental-health-bioactivity-classification"
REMOTE="https://github.com/${REPO_OWNER}/${REPO_NAME}.git"

cd "$(dirname "$0")"

echo "Repository: ${REMOTE}"

# Create repo via GitHub API if token is provided
if [ -n "$GITHUB_TOKEN" ]; then
  echo "Creating GitHub repository (if it does not exist)..."
  curl -sS -X POST \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/user/repos \
    -d "{\"name\":\"${REPO_NAME}\",\"description\":\"ML bioactivity classifiers for mental health drug targets\",\"private\":false}" \
    > /dev/null 2>&1 || true
fi

echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "Done! View at: https://github.com/${REPO_OWNER}/${REPO_NAME}"
