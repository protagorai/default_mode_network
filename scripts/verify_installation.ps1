# SDMN Installation Verification Script (PowerShell)
# This script verifies that the SDMN package has been properly installed on Windows

param(
    [switch]$Help
)

if ($Help) {
    Write-Host "SDMN Installation Verification Script (PowerShell)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This script verifies that the SDMN package has been properly"
    Write-Host "installed and can be imported correctly."
    Write-Host ""
    Write-Host "Usage: .\scripts\verify_installation.ps1 [-Help]"
    Write-Host ""
    Write-Host "Tests performed:"
    Write-Host "  • Basic package import"
    Write-Host "  • Component imports"
    Write-Host "  • CLI availability"
    Write-Host "  • Basic functionality"
    exit 0
}

function Write-TestResult {
    param($TestName, $Success, $Message = "")
    
    if ($Success) {
        Write-Host "[SUCCESS] $TestName" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] $TestName" -ForegroundColor Red
        if ($Message) {
            Write-Host "   $Message" -ForegroundColor Yellow
        }
    }
}

function Write-Info {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Test-BasicImport {
    Write-Host "[TEST] Testing basic SDMN package import..." -ForegroundColor Cyan
    
    try {
        $result = python -c "import sdmn; print(f'SDMN version: {sdmn.__version__}')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-TestResult "Basic package import" $true
            Write-Host "   $result" -ForegroundColor Gray
            return $true
        } else {
            Write-TestResult "Basic package import" $false "Failed to import SDMN package"
            Write-Info "Make sure you have installed the package with:"
            Write-Host "  poetry install --with dev" -ForegroundColor Gray
            Write-Host "  # or" -ForegroundColor Gray
            Write-Host "  pip install -e ." -ForegroundColor Gray
            return $false
        }
    } catch {
        Write-TestResult "Basic package import" $false $_.Exception.Message
        return $false
    }
}

function Test-ComponentImports {
    Write-Host "[TEST] Testing component imports..." -ForegroundColor Cyan
    
    $importTest = @"
try:
    from sdmn.core import SimulationEngine, SimulationConfig
    from sdmn.neurons import LIFNeuron, NeuronType
    from sdmn.networks import NetworkBuilder, NetworkTopology
    from sdmn.probes import VoltageProbe, SpikeProbe
    print('SUCCESS')
except ImportError as e:
    print(f'ERROR: {e}')
except Exception as e:
    print(f'ERROR: {e}')
"@
    
    try {
        $result = python -c $importTest 2>$null
        if ($result -eq "SUCCESS") {
            Write-TestResult "Component imports" $true
            return $true
        } else {
            Write-TestResult "Component imports" $false $result
            return $false
        }
    } catch {
        Write-TestResult "Component imports" $false $_.Exception.Message
        return $false
    }
}

function Test-CLIAvailability {
    Write-Host "[TEST] Testing CLI availability..." -ForegroundColor Cyan
    
    try {
        $result = python -c "import sdmn.cli; print('SUCCESS')" 2>$null
        if ($result -eq "SUCCESS") {
            Write-TestResult "CLI availability" $true
            return $true
        } else {
            Write-TestResult "CLI availability" $false "CLI import failed"
            return $false
        }
    } catch {
        Write-TestResult "CLI availability" $false $_.Exception.Message
        return $false
    }
}

function Test-PackageFunctionality {
    Write-Host "[TEST] Testing basic functionality..." -ForegroundColor Cyan
    
    $functionalityTest = @"
try:
    from sdmn.core import SimulationConfig, TimeManager
    from sdmn.neurons import NeuronType, LIFParameters
    
    # Test creating configuration
    config = SimulationConfig(dt=0.1, max_time=10.0, enable_logging=False)
    print('SimulationConfig created')
    
    # Test creating time manager
    time_mgr = TimeManager(dt=0.1, max_time=10.0)
    print('TimeManager created')
    
    # Test creating neuron parameters
    params = LIFParameters(tau_m=20.0)
    print('LIFParameters created')
    
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {e}')
"@
    
    try {
        $result = python -c $functionalityTest 2>$null
        if ($result -match "SUCCESS") {
            Write-TestResult "Basic functionality" $true
            # Show intermediate results
            $lines = $result -split "`n"
            foreach ($line in $lines) {
                if ($line -ne "SUCCESS" -and $line.Trim() -ne "") {
                    Write-Host "   ✓ $line" -ForegroundColor Gray
                }
            }
            return $true
        } else {
            Write-TestResult "Basic functionality" $false $result
            return $false
        }
    } catch {
        Write-TestResult "Basic functionality" $false $_.Exception.Message
        return $false
    }
}

function Test-WindowsSpecific {
    Write-Host "[TEST] Testing Windows-specific features..." -ForegroundColor Cyan
    
    $passed = 0
    $total = 3
    
    # Test Python in PATH
    try {
        $pythonVersion = python --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $pythonVersion) {
            Write-Host "   ✓ Python in PATH: $pythonVersion" -ForegroundColor Gray
            $passed++
        } else {
            # Try py command as fallback
            $pythonVersion = py --version 2>$null
            if ($LASTEXITCODE -eq 0 -and $pythonVersion) {
                Write-Host "   ✓ Python in PATH (via py): $pythonVersion" -ForegroundColor Gray
                $passed++
            } else {
                Write-Host "   ✗ Python not in PATH" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host "   ✗ Python check failed" -ForegroundColor Red
    }
    
    # Test Poetry availability
    try {
        $poetryVersion = poetry --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✓ Poetry available: $poetryVersion" -ForegroundColor Gray
            $passed++
        } else {
            Write-Host "   ✗ Poetry not available" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   ✗ Poetry check failed" -ForegroundColor Yellow
    }
    
    # Test Docker/Podman availability
    $containerEngine = $null
    try {
        docker --version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $containerEngine = "Docker"
        }
    } catch {}
    
    if (-not $containerEngine) {
        try {
            podman --version 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $containerEngine = "Podman"
            }
        } catch {}
    }
    
    if ($containerEngine) {
        Write-Host "   ✓ Container engine available: $containerEngine" -ForegroundColor Gray
        $passed++
    } else {
        Write-Host "   ✗ No container engine found (Docker/Podman)" -ForegroundColor Yellow
    }
    
    Write-TestResult "Windows environment" ($passed -ge 2) "$passed/$total checks passed"
    return ($passed -ge 2)
}

# Main execution
function Main {
    Write-Host "SDMN Installation Verification (PowerShell)" -ForegroundColor Cyan
    Write-Host "=" * 50
    Write-Host ""
    
    # Check if we're in the right directory
    if (-not (Test-Path "pyproject.toml")) {
        Write-Host "[ERROR] Please run from project root (pyproject.toml not found)" -ForegroundColor Red
        exit 1
    }
    
    # Run tests
    $tests = @(
        { Test-BasicImport },
        { Test-ComponentImports },
        { Test-CLIAvailability },
        { Test-PackageFunctionality },
        { Test-WindowsSpecific }
    )
    
    $passed = 0
    foreach ($test in $tests) {
        if (& $test) {
            $passed++
        }
        Write-Host ""
    }
    
    Write-Host "=" * 50
    Write-Host "Tests passed: $passed/$($tests.Count)" -ForegroundColor $(if ($passed -eq $tests.Count) { "Green" } else { "Yellow" })
    
    if ($passed -eq $tests.Count) {
        Write-Host "[SUCCESS] SDMN package installation verified!" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now use the package:" -ForegroundColor Blue
        Write-Host "  python -m sdmn info" -ForegroundColor Gray
        Write-Host "  python -m sdmn simulate --help" -ForegroundColor Gray
        Write-Host "  python examples\quickstart_simulation.py" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Windows-specific usage:" -ForegroundColor Blue
        Write-Host "  .\activate_sdmn.bat              # Activate environment" -ForegroundColor Gray
        Write-Host "  .\scripts\build.bat              # Build container" -ForegroundColor Gray
        Write-Host "  .\scripts\run.bat shell          # Run container shell" -ForegroundColor Gray
    } else {
        Write-Host "[ERROR] Installation verification failed!" -ForegroundColor Red
        Write-Host "Please check the error messages above." -ForegroundColor Yellow
        
        Write-Host ""
        Write-Host "Common solutions:" -ForegroundColor Blue
        Write-Host "  • Re-run setup: .\scripts\setup_development.bat" -ForegroundColor Gray
        Write-Host "  • Check Python: python --version" -ForegroundColor Gray
        Write-Host "  • Check Poetry: poetry --version" -ForegroundColor Gray
        Write-Host "  • Reinstall package: poetry install --with dev" -ForegroundColor Gray
    }
    
    return ($passed -eq $tests.Count)
}

# Run main function
$success = Main
if (-not $success) {
    exit 1
}
