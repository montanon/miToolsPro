#!/bin/bash
# Quick release script for miToolsPro

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if version type is provided
if [ $# -eq 0 ]; then
    print_error "Please specify version type: major, minor, patch, or prerelease"
    echo "Usage: $0 <version_type> [--dry-run]"
    exit 1
fi

VERSION_TYPE=$1
DRY_RUN=""

if [ "$2" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
    print_warning "Running in DRY RUN mode"
fi

print_status "Starting $VERSION_TYPE release for miToolsPro"

# Check if working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    print_error "Working directory is not clean. Please commit or stash changes."
    exit 1
fi
print_success "Working directory is clean"

# Run tests
print_status "Running tests..."
if ! uv run pytest; then
    print_error "Tests failed. Aborting release."
    exit 1
fi
print_success "Tests passed"

if [ -z "$DRY_RUN" ]; then
    # Bump version and update changelog
    print_status "Bumping $VERSION_TYPE version..."
    uv run cz bump --increment $VERSION_TYPE
    
    NEW_VERSION=$(uv run cz version)
    print_success "Version bumped to $NEW_VERSION"
    
    # Build package
    print_status "Building package..."
    uv build
    print_success "Package built"
    
    # Create git tag (commitizen should handle this, but double-check)
    print_status "Ensuring git tag exists..."
    git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION" 2>/dev/null || echo "Tag already exists"
    
    # Push changes and tags
    print_status "Pushing changes and tags..."
    git push origin HEAD
    git push origin --tags
    
    # Create GitHub release if gh CLI is available
    if command -v gh &> /dev/null; then
        print_status "Creating GitHub release..."
        # Get changelog for this version
        CHANGELOG=$(uv run cz changelog --dry-run --incremental | head -20)
        gh release create "v$NEW_VERSION" --title "Release v$NEW_VERSION" --notes "$CHANGELOG"
        print_success "GitHub release created"
    else
        print_warning "GitHub CLI not found. Skipping GitHub release creation."
        print_status "You can create a release manually at: https://github.com/your-org/miToolsPro/releases/new"
    fi
    
    # Ask about PyPI publishing
    read -p "Publish to PyPI? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Publishing to PyPI..."
        uv publish
        print_success "Published to PyPI"
    else
        print_warning "Skipped PyPI publishing"
        print_status "To publish later, run: uv publish"
    fi
    
    print_success "🎉 Release v$NEW_VERSION completed!"
    
else
    print_status "DRY RUN: Would bump $VERSION_TYPE version"
    print_status "DRY RUN: Would build package"
    print_status "DRY RUN: Would create GitHub release"
    print_status "DRY RUN: Would ask about PyPI publishing"
    print_success "🔍 Dry run completed"
fi

echo
print_status "Release workflow finished!"
echo "Commands you can now use:"
echo "  cz commit                    # Create conventional commit"
echo "  cz changelog                 # View changelog"
echo "  cz bump --increment patch    # Bump patch version"
echo "  uv run python scripts/release.py patch  # Full release workflow"