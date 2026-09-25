import ast
from pathlib import Path
import unittest


class ArchitectureDependencyTest(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SOURCE_ROOT = PROJECT_ROOT / "tonyhar"
    PACKAGES = {
        "agent",
        "conversation",
        "rag",
        "resilience",
        "tooling",
        "tools",
        "web",
    }

    def test_internal_packages_have_no_dependency_cycle(self):
        graph = {package: set() for package in self.PACKAGES}

        for package in self.PACKAGES:
            for path in (self.SOURCE_ROOT / package).rglob("*.py"):
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
                        parts = target.split(".")
                        if len(parts) < 2 or parts[0] != "tonyhar":
                            continue
                        target_package = parts[1]
                        if (
                            target_package in self.PACKAGES
                            and target_package != package
                        ):
                            graph[package].add(target_package)

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

    def test_legacy_top_level_packages_have_no_python_modules(self):
        for package in {
            "conversation",
            "core",
            "rag",
            "resilience",
            "tooling",
            "tools",
        }:
            legacy_root = self.PROJECT_ROOT / package
            self.assertEqual(
                list(legacy_root.rglob("*.py")) if legacy_root.exists() else [],
                [],
                f"旧顶层包仍包含 Python 模块: {package}",
            )


if __name__ == "__main__":
    unittest.main()
