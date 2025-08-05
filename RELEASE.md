# Release Guide for miToolsPro

This document provides a comprehensive guide to version control, conventional commits, and release management for miToolsPro.

## Table of Contents

- [Overview](#overview)
- [Version Control Strategy](#version-control-strategy)
- [Conventional Commits](#conventional-commits)
- [Release Workflow](#release-workflow)
- [Tools and Scripts](#tools-and-scripts)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

## Overview

miToolsPro uses a modern, automated release system that combines:

- **Semantic Versioning** (SemVer) for predictable version numbers
- **Conventional Commits** for structured commit messages
- **Automated Changelog Generation** from commit history
- **Integrated Release Workflow** with GitHub and PyPI publishing
- **uv Package Manager** for modern Python dependency management

### Key Benefits

- ✅ **Automated versioning** based on commit types
- ✅ **Generated changelogs** from commit messages
- ✅ **Consistent release process** with minimal manual steps
- ✅ **GitHub integration** for releases and tags
- ✅ **PyPI publishing** automation
- ✅ **Developer-friendly** tools and scripts

## Version Control Strategy

### Semantic Versioning (SemVer)

We follow [Semantic Versioning 2.0.0](https://semver.org/) with the format `MAJOR.MINOR.PATCH`:

- **MAJOR** (1.0.0 → 2.0.0): Breaking changes that require user code updates
- **MINOR** (1.0.0 → 1.1.0): New features that are backwards compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes and patches

### Pre-release Versions

For pre-release versions, we use:
- **Alpha**: `1.0.0a1`, `1.0.0a2` - Early development, unstable
- **Beta**: `1.0.0b1`, `1.0.0b2` - Feature complete, testing phase
- **Release Candidate**: `1.0.0rc1`, `1.0.0rc2` - Final testing before release

### Version Sources

- **Single Source of Truth**: Git tags via `hatch-vcs`
- **Dynamic Loading**: Version read from package metadata at runtime
- **Development Versions**: Automatic `.devN+hash` suffixes for unreleased commits

```python
# How version is determined
import mitoolspro
print(mitoolspro.__version__)  # e.g., "1.0.0" or "1.0.1.dev5+g123abc"
```

## Conventional Commits

### Format

Conventional commits follow this structure:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit Types

| Type | Description | Version Impact | Changelog Section |
|------|-------------|----------------|-------------------|
| `feat` | New feature | Minor bump | Features |
| `fix` | Bug fix | Patch bump | Bug Fixes |
| `docs` | Documentation | No bump | Documentation |
| `style` | Code style (formatting, etc.) | No bump | - |
| `refactor` | Code refactoring | No bump | - |
| `perf` | Performance improvements | Patch bump | Performance |
| `test` | Adding or updating tests | No bump | - |
| `chore` | Maintenance tasks | No bump | - |
| `ci` | CI/CD changes | No bump | - |
| `build` | Build system changes | No bump | - |

### Breaking Changes

For breaking changes, use `!` or `BREAKING CHANGE:` footer:

```bash
# Using ! suffix
git commit -m "feat!: redesign API for better type safety"

# Using footer
git commit -m "feat: add new authentication method

BREAKING CHANGE: The old auth() method has been removed in favor of authenticate()"
```

### Examples

```bash
# New features
git commit -m "feat: add K-means clustering algorithm"
git commit -m "feat(plotting): implement scatter plot with regression line"

# Bug fixes
git commit -m "fix: resolve memory leak in matplotlib wrapper"
git commit -m "fix(llms): handle API timeout errors gracefully"

# Documentation
git commit -m "docs: add examples for regression analysis"
git commit -m "docs(api): update docstrings for plotting module"

# Maintenance
git commit -m "chore: update dependencies to latest versions"
git commit -m "test: add unit tests for database utilities"
```

### Using Interactive Commit Tool

```bash
# Interactive conventional commit (recommended)
uv run cz commit

# This will prompt you for:
# 1. Commit type (feat, fix, docs, etc.)
# 2. Scope (optional, e.g., plotting, llms)
# 3. Description
# 4. Body (optional)
# 5. Breaking changes (if any)
# 6. Issues closed (if any)
```

## Release Workflow

### Automated Release Process

1. **Commit Analysis**: Commitizen analyzes commits since last release
2. **Version Calculation**: Determines version bump based on commit types
3. **Changelog Generation**: Creates changelog entries from commits
4. **Version Bump**: Updates version and creates git tag
5. **Package Building**: Builds distribution packages with uv
6. **GitHub Release**: Creates release with generated notes
7. **PyPI Publishing**: Uploads package to Python Package Index

### Release Types

#### Patch Release (1.0.0 → 1.0.1)
- Bug fixes and small improvements
- No new features or breaking changes

```bash
./scripts/release.sh patch
# or
uv run python scripts/release.py patch
```

#### Minor Release (1.0.0 → 1.1.0)
- New features that are backwards compatible
- May include bug fixes

```bash
./scripts/release.sh minor
```

#### Major Release (1.0.0 → 2.0.0)
- Breaking changes that affect user code
- May include new features and bug fixes

```bash
./scripts/release.sh major
```

#### Pre-release
- Testing versions before final release

```bash
./scripts/release.sh prerelease
```

### Manual Release Steps

If you need to perform releases manually:

```bash
# 1. Ensure clean working directory
git status

# 2. Run tests
uv run pytest

# 3. Bump version and update changelog
uv run cz bump --increment patch

# 4. Build package
uv build

# 5. Create GitHub release
gh release create v$(uv run cz version) --generate-notes

# 6. Publish to PyPI
uv publish
```

## Tools and Scripts

### Release Scripts

#### `scripts/release.sh` (Recommended)
Interactive bash script with colored output and safety checks.

```bash
# Usage
./scripts/release.sh <version_type> [--dry-run]

# Examples
./scripts/release.sh patch           # Patch release
./scripts/release.sh minor --dry-run # Preview minor release
```

**Features:**
- ✅ Colored output for better readability
- ✅ Safety checks (clean working directory, tests)
- ✅ Interactive PyPI publishing confirmation
- ✅ Automatic GitHub release creation
- ✅ Dry run mode for testing

#### `scripts/release.py`
Full-featured Python script with advanced options.

```bash
# Usage
uv run python scripts/release.py <version_type> [options]

# Options
--dry-run      # Preview changes without executing
--skip-tests   # Skip test execution
--skip-pypi    # Skip PyPI publishing
```

### Commitizen Commands

```bash
# Interactive commit creation
uv run cz commit

# Check current version
uv run cz version

# Bump version manually
uv run cz bump --increment patch

# Generate changelog
uv run cz changelog

# Validate commit message
uv run cz check --rev-range HEAD~1..HEAD
```

### Pre-commit Hooks

Pre-commit hooks ensure code quality and commit message format:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files

# Update hook versions
uv run pre-commit autoupdate
```

## Common Tasks

### Starting Development

```bash
# Clone repository
git clone <repository-url>
cd miToolsPro

# Install dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install

# Verify setup
uv run cz version
```

### Making Changes

```bash
# Create feature branch
git checkout -b feature/new-clustering-algorithm

# Make your changes
# ... edit files ...

# Add and commit with conventional format
git add .
uv run cz commit  # Interactive commit

# Push branch
git push origin feature/new-clustering-algorithm

# Create pull request (use GitHub UI or gh CLI)
gh pr create --title "feat: add new clustering algorithm"
```

### Preparing for Release

```bash
# Ensure you're on main branch
git checkout main
git pull origin main

# Check what's changed since last release
git log $(git describe --tags --abbrev=0)..HEAD --oneline

# Preview the release
./scripts/release.sh patch --dry-run

# Perform the release
./scripts/release.sh patch
```

### Hotfix Release

```bash
# Create hotfix branch from main
git checkout main
git checkout -b hotfix/critical-bug-fix

# Make minimal fix
# ... edit files ...

# Commit fix
git commit -m "fix: resolve critical security vulnerability"

# Merge to main
git checkout main
git merge hotfix/critical-bug-fix

# Release immediately
./scripts/release.sh patch
```

## Troubleshooting

### Common Issues

#### "Working directory is not clean"
```bash
# Check what's uncommitted
git status

# Stash or commit changes
git stash  # or git add . && git commit -m "wip: save progress"
```

#### "Tests failed"
```bash
# Run tests to see failures
uv run pytest -v

# Fix issues and retry
./scripts/release.sh patch
```

#### "Version bump failed"
```bash
# Check commitizen configuration
uv run cz info

# Manually bump version
uv run cz bump --increment patch --dry-run  # preview
uv run cz bump --increment patch           # execute
```

#### "GitHub release failed"
```bash
# Check GitHub CLI authentication
gh auth status

# Login if needed
gh auth login

# Create release manually
gh release create v$(uv run cz version) --generate-notes
```

#### "PyPI publishing failed"
```bash
# Check authentication
uv publish --dry-run

# Configure PyPI token
export UV_PUBLISH_TOKEN=your_pypi_token

# Or use interactive login
uv publish --username __token__ --password your_pypi_token
```

### Version Mismatches

If you encounter version inconsistencies:

```bash
# Check current version sources
uv run python -c "import mitoolspro; print(mitoolspro.__version__)"
uv run cz version
git describe --tags

# Rebuild package if needed
uv build --clean
```

### Rollback a Release

If you need to rollback a problematic release:

```bash
# Delete the tag locally and remotely
git tag -d v1.2.3
git push origin :refs/tags/v1.2.3

# Delete GitHub release
gh release delete v1.2.3

# Revert commits if necessary
git revert <commit-hash>
```

## Configuration Files

### `pyproject.toml` - Commitizen Configuration
```toml
[tool.commitizen]
name = "cz_conventional_commits"
tag_format = "v$version"
version_scheme = "pep440"
version_provider = "scm"
update_changelog_on_bump = true
major_version_zero = false
changelog_file = "CHANGELOG.md"
```

### `.pre-commit-config.yaml` - Pre-commit Hooks
```yaml
repos:
  - repo: https://github.com/commitizen-tools/commitizen
    rev: v3.29.0
    hooks:
      - id: commitizen
        stages: [commit-msg]
```

## Best Practices

### Development Workflow
1. **Always use conventional commits** for clear history
2. **Run tests before committing** to catch issues early
3. **Use feature branches** for non-trivial changes
4. **Keep commits atomic** - one logical change per commit
5. **Write descriptive commit messages** that explain the "why"

### Release Management
1. **Test releases in staging** before production
2. **Use dry-run mode** to preview changes
3. **Document breaking changes** clearly in commit messages
4. **Follow semantic versioning** strictly
5. **Keep changelog updated** through conventional commits

### Security Considerations
1. **Never commit secrets** or API keys
2. **Use environment variables** for sensitive configuration
3. **Review dependencies** regularly for vulnerabilities
4. **Sign releases** with GPG keys (optional but recommended)

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Release
on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  release:
    if: startsWith(github.ref, 'refs/tags/')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      
      - name: Set up Python
        run: uv python install 3.12
      
      - name: Install dependencies
        run: uv sync --group dev
      
      - name: Run tests
        run: uv run pytest
      
      - name: Build package
        run: uv build
      
      - name: Publish to PyPI
        run: uv publish
        env:
          UV_PUBLISH_TOKEN: ${{ secrets.PYPI_TOKEN }}
      
      - name: Create GitHub Release
        run: gh release create ${{ github.ref_name }} --generate-notes
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Summary

This release system provides:

- **Automated version management** based on commit conventions
- **Generated documentation** through changelogs
- **Streamlined release process** with safety checks
- **Integration with modern tools** (uv, GitHub, PyPI)
- **Developer-friendly workflows** with clear guidelines

For questions or issues, refer to the project's documentation or create an issue in the repository.

**Happy releasing! 🚀**