#!/usr/bin/env python3
"""
Recursively analyze local Python file dependencies starting from a specific file.
Only analyzes files that are actually imported (not all files in the directory).
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, deque
import argparse


class RecursiveDependencyAnalyzer:
    def __init__(self, root_path='.'):
        self.root_path = Path(root_path).resolve()
        
        # Maps: module_name -> set of modules it imports
        self.dependencies = defaultdict(set)
        # Maps: module_name -> set of modules that import it
        self.dependents = defaultdict(set)
        # Maps: module_name -> file_path
        self.module_files = {}
        # Track what we've already analyzed
        self.analyzed = set()
        
    def path_to_module_name(self, file_path):
        """Convert file path to module name relative to root."""
        try:
            rel_path = file_path.resolve().relative_to(self.root_path)
        except ValueError:
            # File is outside root_path, use absolute module name
            return file_path.stem
        
        # Remove .py extension
        if rel_path.name == '__init__.py':
            # For __init__.py files, use the directory name
            module_parts = rel_path.parts[:-1]
        else:
            module_parts = rel_path.parts[:-1] + (rel_path.stem,)
        
        return '.'.join(module_parts) if module_parts else rel_path.stem
    
    def module_name_to_path(self, module_name, from_file=None):
        """Try to resolve a module name to a file path."""
        if from_file:
            from_dir = from_file.parent
        else:
            from_dir = self.root_path
        
        # Handle relative imports if from_file is provided
        if module_name.startswith('.'):
            if not from_file:
                return None
            
            # Count leading dots
            level = 0
            for char in module_name:
                if char == '.':
                    level += 1
                else:
                    break
            
            # Get the current module's path components
            current_module = self.path_to_module_name(from_file)
            if '.' in current_module:
                base_parts = current_module.split('.')[:-1]
            else:
                base_parts = []
            
            # Go up 'level-1' directories
            if level > 1:
                base_parts = base_parts[:-(level-1)] if level-1 < len(base_parts) else []
            
            # Add the module part (everything after the dots)
            remaining_module = module_name[level:]
            if remaining_module:
                module_parts = base_parts + remaining_module.split('.')
            else:
                module_parts = base_parts
                
            # Convert back to file path
            potential_path = self.root_path / Path(*module_parts)
        else:
            # Absolute import - look relative to root or from current directory
            module_parts = module_name.split('.')
            
            # Try relative to root first
            potential_path = self.root_path / Path(*module_parts)
            
            # If not found, try relative to current file's directory
            if not potential_path.with_suffix('.py').exists() and not (potential_path / '__init__.py').exists():
                if from_file:
                    potential_path = from_file.parent / Path(*module_parts)
        
        # Check if it exists as a .py file
        py_file = potential_path.with_suffix('.py')
        if py_file.exists():
            return py_file
        
        # Check if it exists as a package (__init__.py)
        init_file = potential_path / '__init__.py'
        if init_file.exists():
            return init_file
        
        return None
    
    def extract_imports(self, file_path):
        """Extract all imports from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return set()
        
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                elif node.level > 0:
                    # Relative import without module (from . import something)
                    imports.add('.' * node.level)
        
        return imports
    
    def analyze_file(self, file_path):
        """Analyze a single file and return its local dependencies."""
        file_path = Path(file_path).resolve()
        module_name = self.path_to_module_name(file_path)
        
        if module_name in self.analyzed:
            return []
        
        self.analyzed.add(module_name)
        self.module_files[module_name] = file_path
        
        imports = self.extract_imports(file_path)
        local_deps = []
        
        for imp in imports:
            dep_path = self.module_name_to_path(imp, file_path)
            if dep_path and dep_path.exists():
                dep_module = self.path_to_module_name(dep_path)
                self.dependencies[module_name].add(dep_module)
                self.dependents[dep_module].add(module_name)
                local_deps.append(dep_path)
        
        return local_deps
    
    def analyze_recursive(self, start_file):
        """Recursively analyze dependencies starting from a file."""
        start_file = Path(start_file).resolve()
        if not start_file.exists():
            print(f"Error: File {start_file} does not exist")
            return
        
        to_analyze = deque([start_file])
        
        while to_analyze:
            current_file = to_analyze.popleft()
            local_deps = self.analyze_file(current_file)
            
            # Add newly found dependencies to the queue
            for dep_file in local_deps:
                dep_module = self.path_to_module_name(dep_file)
                if dep_module not in self.analyzed:
                    to_analyze.append(dep_file)
    
    def print_dependencies(self, module_name=None):
        """Print dependencies for a specific module or all analyzed modules."""
        if module_name:
            if module_name not in self.module_files:
                print(f"Module '{module_name}' not found in analyzed modules")
                return
            modules_to_print = [module_name]
        else:
            modules_to_print = sorted(self.module_files.keys())
        
        for mod in modules_to_print:
            deps = self.dependencies.get(mod, set())
            print(f"\n{mod}:")
            print(f"  File: {self.module_files[mod]}")
            if deps:
                print(f"  Dependencies: {', '.join(sorted(deps))}")
            else:
                print(f"  Dependencies: None")
    
    def print_dependents(self, module_name=None):
        """Print dependents for a specific module or all analyzed modules."""
        if module_name:
            if module_name not in self.module_files:
                print(f"Module '{module_name}' not found in analyzed modules")
                return
            modules_to_print = [module_name]
        else:
            modules_to_print = sorted(self.module_files.keys())
        
        for mod in modules_to_print:
            deps = self.dependents.get(mod, set())
            print(f"\n{mod}:")
            print(f"  File: {self.module_files[mod]}")
            if deps:
                print(f"  Used by: {', '.join(sorted(deps))}")
            else:
                print(f"  Used by: None (leaf/entry point)")
    
    def print_dependency_tree(self, module_name, max_depth=None, _current_depth=0, _visited=None):
        """Print dependency tree for a specific module."""
        if _visited is None:
            _visited = set()
        
        if module_name in _visited:
            print("  " * _current_depth + f"{module_name} (circular)")
            return
        
        if max_depth is not None and _current_depth >= max_depth:
            print("  " * _current_depth + f"{module_name} ...")
            return
        
        _visited.add(module_name)
        print("  " * _current_depth + module_name)
        
        deps = self.dependencies.get(module_name, set())
        for dep in sorted(deps):
            self.print_dependency_tree(dep, max_depth, _current_depth + 1, _visited.copy())
    
    def find_circular_dependencies(self):
        """Find circular dependencies in analyzed modules."""
        def has_cycle(start, current, path, visited):
            if current in path:
                cycle_start = path.index(current)
                return path[cycle_start:] + [current]
            
            if current in visited:
                return None
            
            visited.add(current)
            path.append(current)
            
            for dep in self.dependencies.get(current, set()):
                result = has_cycle(start, dep, path.copy(), visited)
                if result:
                    return result
            
            return None
        
        cycles = []
        visited_global = set()
        
        for module in self.module_files:
            if module not in visited_global:
                cycle = has_cycle(module, module, [], set())
                if cycle:
                    cycles.append(cycle)
                    visited_global.update(cycle)
        
        return cycles
    
    def print_summary(self):
        """Print a summary of the analysis."""
        print(f"\nSummary:")
        print(f"  Total modules analyzed: {len(self.module_files)}")
        print(f"  Modules with dependencies: {len([m for m in self.dependencies if self.dependencies[m]])}")
        print(f"  Modules with dependents: {len([m for m in self.dependents if self.dependents[m]])}")
        
        # Find entry points (modules with no dependents)
        entry_points = [m for m in self.module_files if not self.dependents.get(m)]
        if entry_points:
            print(f"  Entry points: {', '.join(sorted(entry_points))}")
    
    def create_shar(self, output_file=None, include_subdirs=True):
        """Create a shell archive (shar) of all analyzed Python files."""
        import time
        import getpass
        import socket
        
        if not output_file:
            # Generate filename based on entry point
            entry_points = [m for m in self.module_files if not self.dependents.get(m)]
            if entry_points:
                base_name = entry_points[0].replace('.', '_')
            else:
                base_name = "python_deps"
            output_file = f"{base_name}_bundle.shar"
        
        # Collect all files and their relative paths
        files_to_include = []
        for module_name, file_path in self.module_files.items():
            try:
                # Try to get relative path from root
                rel_path = file_path.relative_to(self.root_path)
            except ValueError:
                # File is outside root, use just the filename
                rel_path = file_path.name
            
            files_to_include.append((file_path, rel_path))
        
        # Sort by path for consistent output
        files_to_include.sort(key=lambda x: str(x[1]))
        
        with open(output_file, 'w') as shar:
            # Write shar header
            shar.write("#!/bin/sh\n")
            shar.write("# This is a shell archive created by Python dependency analyzer\n")
            shar.write(f"# Created on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            shar.write(f"# Created by: {getpass.getuser()}@{socket.gethostname()}\n")
            shar.write(f"# Contains {len(files_to_include)} Python files\n")
            shar.write("#\n")
            shar.write("# To extract, run: sh {}\n".format(output_file))
            shar.write("#\n\n")
            
            # List of files
            shar.write("echo 'Extracting Python dependency bundle...'\n")
            shar.write("echo 'Files included:'\n")
            for _, rel_path in files_to_include:
                shar.write(f"echo '  {rel_path}'\n")
            shar.write("echo ''\n\n")
            
            # Create directories if needed
            if include_subdirs:
                dirs_needed = set()
                for _, rel_path in files_to_include:
                    parent = rel_path.parent
                    while parent != Path('.'):
                        dirs_needed.add(parent)
                        parent = parent.parent
                
                if dirs_needed:
                    shar.write("# Create directories\n")
                    for dir_path in sorted(dirs_needed):
                        shar.write(f"mkdir -p '{dir_path}'\n")
                    shar.write("\n")
            
            # Extract each file
            for file_path, rel_path in files_to_include:
                shar.write(f"# Extracting {rel_path}\n")
                shar.write(f"cat > '{rel_path}' << 'EOF_{rel_path.name.upper().replace('.', '_')}'\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Ensure content ends with newline
                        if content and not content.endswith('\n'):
                            content += '\n'
                        shar.write(content)
                except UnicodeDecodeError:
                    shar.write(f"# Warning: Could not read {file_path} as UTF-8\n")
                
                shar.write(f"EOF_{rel_path.name.upper().replace('.', '_')}\n\n")
            
            # Footer
            shar.write("echo 'Extraction complete!'\n")
            shar.write("echo 'Files extracted:'\n")
            shar.write("find . -name '*.py' -type f | sort\n")
        
        # Make the shar executable
        os.chmod(output_file, 0o755)
        
        print(f"Shell archive created: {output_file}")
        print(f"To extract: sh {output_file}")
        print(f"Files included: {len(files_to_include)}")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Recursively analyze local Python dependencies from a starting file')
    parser.add_argument('file', help='Python file to start analysis from')
    parser.add_argument('-r', '--root', default='.', help='Root directory for resolving imports (default: current directory)')
    parser.add_argument('-m', '--module', help='Focus on specific module')
    parser.add_argument('-d', '--dependencies', action='store_true', help='Show dependencies only')
    parser.add_argument('-p', '--dependents', action='store_true', help='Show dependents only')
    parser.add_argument('-t', '--tree', action='store_true', help='Show dependency tree')
    parser.add_argument('-c', '--circular', action='store_true', help='Find circular dependencies')
    parser.add_argument('-s', '--summary', action='store_true', help='Show summary')
    parser.add_argument('--shar', nargs='?', const='', help='Create shell archive of all dependencies (optionally specify output filename)')
    parser.add_argument('--max-depth', type=int, help='Maximum depth for tree display')
    
    args = parser.parse_args()
    
    analyzer = RecursiveDependencyAnalyzer(args.root)
    analyzer.analyze_recursive(args.file)
    
    if args.shar is not None:
        output_file = args.shar if args.shar else None
        analyzer.create_shar(output_file)
        return
    
    if args.circular:
        cycles = analyzer.find_circular_dependencies()
        if cycles:
            print("Circular dependencies found:")
            for i, cycle in enumerate(cycles, 1):
                print(f"  {i}. {' -> '.join(cycle)}")
        else:
            print("No circular dependencies found.")
        return
    
    if args.tree:
        if args.module:
            print(f"Dependency tree for {args.module}:")
            analyzer.print_dependency_tree(args.module, args.max_depth)
        else:
            # Show tree for the starting file
            start_module = analyzer.path_to_module_name(Path(args.file))
            print(f"Dependency tree for {start_module}:")
            analyzer.print_dependency_tree(start_module, args.max_depth)
        return
    
    if args.dependencies:
        analyzer.print_dependencies(args.module)
    elif args.dependents:
        analyzer.print_dependents(args.module)
    elif args.summary:
        analyzer.print_summary()
    else:
        # Default: show both dependencies and dependents
        print("=== DEPENDENCIES ===")
        analyzer.print_dependencies(args.module)
        print("\n=== DEPENDENTS ===")
        analyzer.print_dependents(args.module)
        
        if not args.module:
            analyzer.print_summary()


if __name__ == '__main__':
    main()
