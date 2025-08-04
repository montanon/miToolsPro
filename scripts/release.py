#!/usr/bin/env python3
"""Release script for miToolsPro with conventional commits and GitHub integration."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Literal

VersionType = Literal["major", "minor", "patch", "prerelease"]


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    return result


def check_working_directory_clean() -> None:
    """Ensure the working directory is clean."""
    result = run_command("git status --porcelain")
    if result.stdout.strip():
        print("Error: Working directory is not clean. Please commit or stash changes.")
        sys.exit(1)
    print("✓ Working directory is clean")


def run_tests() -> None:
    """Run the test suite."""
    print("Running tests...")
    run_command("uv run pytest")
    print("✓ Tests passed")


def bump_version(version_type: VersionType) -> str:
    """Bump version using commitizen."""
    print(f"Bumping {version_type} version...")
    
    # Use commitizen to bump version and update changelog
    cmd = f"uv run cz bump --increment {version_type}"
    result = run_command(cmd)
    
    # Get the new version
    version_result = run_command("uv run cz version")
    new_version = version_result.stdout.strip()
    
    print(f"✓ Version bumped to {new_version}")
    return new_version


def build_package() -> None:
    """Build the package."""
    print("Building package...")
    run_command("uv build")
    print("✓ Package built")


def create_github_release(version: str, dry_run: bool = False) -> None:
    """Create a GitHub release."""
    print(f"Creating GitHub release for v{version}...")
    
    # Extract changelog for this version
    changelog_cmd = f"uv run cz changelog --dry-run --incremental"
    changelog_result = run_command(changelog_cmd)
    release_notes = changelog_result.stdout.strip()
    
    # Create GitHub release
    gh_cmd = f'gh release create v{version} --title "Release v{version}" --notes "{release_notes}"'
    
    if dry_run:
        print(f"Would run: {gh_cmd}")
    else:
        run_command(gh_cmd)
        print(f"✓ GitHub release v{version} created")


def publish_to_pypi(dry_run: bool = False) -> None:
    """Publish to PyPI."""
    if dry_run:
        print("Would publish to PyPI (dry run)")
        run_command("uv publish --dry-run")
    else:
        print("Publishing to PyPI...")
        run_command("uv publish")
        print("✓ Published to PyPI")


def main() -> None:
    """Main release workflow."""
    parser = argparse.ArgumentParser(description="Release miToolsPro")
    parser.add_argument(
        "version_type",
        choices=["major", "minor", "patch", "prerelease"],
        help="Type of version bump"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true", 
        help="Skip running tests"
    )
    parser.add_argument(
        "--skip-pypi",
        action="store_true",
        help="Skip publishing to PyPI"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting {args.version_type} release for miToolsPro")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
    
    # Pre-flight checks
    check_working_directory_clean()
    
    if not args.skip_tests:
        run_tests()
    
    # Version bump and changelog update
    if not args.dry_run:
        new_version = bump_version(args.version_type)
    else:
        print(f"Would bump {args.version_type} version")
        new_version = "1.0.0"  # Placeholder for dry run
    
    # Build package
    if not args.dry_run:
        build_package()
    else:
        print("Would build package")
    
    # Create GitHub release
    create_github_release(new_version, dry_run=args.dry_run)
    
    # Publish to PyPI
    if not args.skip_pypi:
        publish_to_pypi(dry_run=args.dry_run)
    
    print(f"🎉 Release v{new_version} completed successfully!")
    
    if not args.dry_run:
        print("\nNext steps:")
        print("1. Verify the release on GitHub")
        print("2. Check the package on PyPI")
        print("3. Update any documentation that references the version")


if __name__ == "__main__":
    main()