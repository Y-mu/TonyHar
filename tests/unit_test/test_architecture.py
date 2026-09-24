import ast
from pathlib import Path
import unittest


class ArchitectureDependencyTest(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PACKAGES = {"core", "rag", "tooling", "tools"}

    def test_internal_packages_have_no_dependency_cycle(self):
        graph = {package: set() for package in self.PACKAGES}

        for package in self.PACKAGES:
            for path in (self.PROJECT_ROOT / package).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    targets: list[str] = []
                    if isinstance(node, ast.Import):
                        targets = [alias.name for alias in node.names]
                    elif (
                        isinstance(node, ast.ImportFrom)
                        and node.level == 0
                        and node.module
                    ):
                        targets = [node.module]

                    for target in targets:
                        top_level = target.split(".", 1)[0]
                        if top_level in self.PACKAGES and top_level != package:
                            graph[package].add(top_level)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(package: str, path: tuple[str, ...]) -> None:
            if package in visiting:
                cycle = " -> ".join((*path, package))
                self.fail(f"检测到内部包循环依赖: {cycle}")
            if package in visited:
                return

            visiting.add(package)
            for dependency in graph[package]:
                visit(dependency, (*path, package))
            visiting.remove(package)
            visited.add(package)

        for package in graph:
            visit(package, ())


if __name__ == "__main__":
    unittest.main()
