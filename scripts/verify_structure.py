#!/usr/bin/env python3
"""
SDMN Package Structure Verification

This script verifies the package structure and imports without requiring
the package to be installed. Perfect for CI/CD and initial setup verification.
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict


def analyze_package_structure():
    """Analyze the package structure."""
    print("SDMN Package Structure Verification")
    print("=" * 50)
    
    src_dir = Path("src/sdmn")
    if not src_dir.exists():
        print("[ERROR] Package directory src/sdmn does not exist!")
        return False
    
    # Count files
    python_files = list(src_dir.rglob("*.py"))
    init_files = list(src_dir.rglob("__init__.py"))
    module_files = [f for f in python_files if f.name != "__init__.py"]
    
    print(f"[INFO] Package location: {src_dir}")
    print(f"[INFO] Total Python files: {len(python_files)}")
    print(f"[INFO] Module files: {len(module_files)}")
    print(f"[INFO] __init__.py files: {len(init_files)}")
    
    return True


def verify_import_structure():
    """Verify all imports are absolute."""
    print("\n[CHECK] Import Structure")
    print("-" * 30)
    
    src_dir = Path("src")
    python_files = list(src_dir.rglob("*.py"))
    
    relative_imports = []
    absolute_imports = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            file_has_relative = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level > 0:
                    file_has_relative = True
                    relative_imports.append(str(file_path.relative_to(Path.cwd())))
                    break
            
            if not file_has_relative:
                absolute_imports.append(str(file_path.relative_to(Path.cwd())))
                
        except Exception as e:
            print(f"[WARNING] Could not parse {file_path}: {e}")
    
    print(f"[RESULT] Files with absolute imports: {len(absolute_imports)}")
    print(f"[RESULT] Files with relative imports: {len(relative_imports)}")
    
    if relative_imports:
        print("[ERROR] Files with relative imports found:")
        for file_path in relative_imports:
            print(f"   - {file_path}")
        return False
    
    print("[SUCCESS] All imports are absolute")
    return True


def verify_examples():
    """Verify examples use correct imports."""
    print("\n[CHECK] Example Files")
    print("-" * 30)
    
    examples_dir = Path("examples")
    if not examples_dir.exists():
        print("[WARNING] Examples directory not found")
        return True
    
    python_files = list(examples_dir.glob("*.py"))
    issues = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for old import patterns
            if 'sys.path.insert' in content:
                issues.append(f"{file_path.name}: Still uses sys.path.insert")
            
            if 'from neurons import' in content or 'from core import' in content:
                issues.append(f"{file_path.name}: Uses old import style")
            
            if 'from sdmn.' not in content and 'import sdmn' not in content:
                issues.append(f"{file_path.name}: No sdmn imports found")
                
        except Exception as e:
            issues.append(f"{file_path.name}: Error reading - {e}")
    
    print(f"[INFO] Analyzed {len(python_files)} example files")
    
    if issues:
        print("[ERROR] Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("[SUCCESS] All examples use correct imports")
    return True


def verify_scripts_organization():
    """Verify scripts are properly organized."""
    print("\n[CHECK] Scripts Organization")
    print("-" * 30)
    
    scripts_dir = Path("scripts")
    if not scripts_dir.exists():
        print("[ERROR] Scripts directory not found")
        return False
    
    expected_scripts = [
        "setup_development.sh",
        "setup_production.sh", 
        "build.sh",
        "run.sh",
        "setup.sh",
        "verify_installation.py",
        "verify_structure.py"
    ]
    
    missing = []
    present = []
    
    for script in expected_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            present.append(script)
        else:
            missing.append(script)
    
    print(f"[INFO] Scripts present: {len(present)}")
    print(f"[INFO] Scripts missing: {len(missing)}")
    
    for script in present:
        print(f"   + {script}")
    
    if missing:
        print("[WARNING] Missing scripts:")
        for script in missing:
            print(f"   - {script}")
    
    # Check if old setup scripts are in root
    old_scripts = []
    for old_script in ["setup_development.sh", "setup_production.sh"]:
        if Path(old_script).exists():
            old_scripts.append(old_script)
    
    if old_scripts:
        print("[WARNING] Old setup scripts still in root:")
        for script in old_scripts:
            print(f"   - {script}")
        print("[INFO] These should be moved to scripts/ directory")
    
    return len(missing) == 0


def verify_configuration_files():
    """Verify configuration files are present."""
    print("\n[CHECK] Configuration Files")
    print("-" * 30)
    
    config_files = [
        "pyproject.toml",
        "poetry.lock",
        "Dockerfile",
        ".github/workflows/ci.yml",
        ".gitlab-ci.yml",
        ".pre-commit-config.yaml"
    ]
    
    missing = []
    present = []
    
    for config_file in config_files:
        if Path(config_file).exists():
            present.append(config_file)
        else:
            missing.append(config_file)
    
    print(f"[INFO] Configuration files present: {len(present)}")
    
    for file in present:
        print(f"   + {file}")
    
    if missing:
        print(f"[WARNING] Missing configuration files: {len(missing)}")
        for file in missing:
            print(f"   - {file}")
    
    return len(missing) == 0


def main():
    """Main verification."""
    success = True
    
    # Verify we're in project root
    if not Path("pyproject.toml").exists():
        print("[ERROR] Please run from project root (pyproject.toml not found)")
        return False
    
    # Run all checks
    if not analyze_package_structure():
        success = False
    
    if not verify_import_structure():
        success = False
    
    if not verify_examples():
        success = False
    
    if not verify_scripts_organization():
        success = False
    
    if not verify_configuration_files():
        success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("[SUCCESS] Package structure verification passed!")
        print("")
        print("Package is ready for:")
        print("  • Local development: ./scripts/setup_development.sh")
        print("  • Local production: ./scripts/setup_production.sh")
        print("  • Containerized deployment: ./scripts/build.sh")
        print("  • Interactive setup: ./scripts/setup.sh")
    else:
        print("[ERROR] Package structure verification failed!")
        print("Please fix the issues above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
