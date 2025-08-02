import argparse
from pathlib import Path

from mitoolspro.project import Project
import logging

logger = logging.getLogger(__name__)


def init_project(args):
    project = Project(
        project_name=args.name,
        root=args.root,
        version=args.version,
    )
    logger.info("Initialized project '%s' in %s", args.name, args.root)


def main():
    parser = argparse.ArgumentParser(description="miToolsPro CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    init_parser = subparsers.add_parser("init", help="Initialize a new project")
    init_parser.add_argument("name", help="Name of the project")
    init_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory for the project (default: current directory)",
    )
    init_parser.add_argument(
        "--version",
        default="v0",
        help="Initial version of the project (default: v0)",
    )

    args = parser.parse_args()

    if args.command == "init":
        init_project(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
