param()

# Windows can expose several different `bash.exe` paths. This script records
# which one is usable and marks broken WSL routing as a local setup blocker, not
# as evidence that the repository's Bash scripts are faulty.
$ErrorActionPreference = "Continue"
Set-StrictMode -Version Latest

function Write-Result {
    param([string]$Status, [string]$Detail)
    Write-Host "${Status}: ${Detail}"
}

$candidates = @()
$systemBash = Join-Path $env:WINDIR "System32\bash.exe"
if (Test-Path $systemBash) { $candidates += $systemBash }
$gitBash = "C:\Program Files\Git\bin\bash.exe"
if (Test-Path $gitBash) { $candidates += $gitBash }
$pathBash = Get-Command bash -ErrorAction SilentlyContinue
if ($null -ne $pathBash) { $candidates += $pathBash.Source }
$candidates = $candidates | Select-Object -Unique

if (-not $candidates) {
    Write-Result "LOCAL_BASH_BLOCKED_BY_WINDOWS_WSL_SETUP" "No bash executable found. Install Git Bash, fix WSL, use PowerShell scripts, or rely on GitHub Actions Ubuntu Bash verification."
    exit 0
}

Write-Result "INFO" "Bash candidates: $($candidates -join '; ')"

$usable = $false
foreach ($candidate in $candidates) {
    try {
        $output = & $candidate -lc "echo BASH_OK && uname -s" 2>&1
        $exitCode = $LASTEXITCODE
        $outputText = ($output | Out-String).Trim()
        if ($exitCode -eq 0 -and $outputText -match "BASH_OK") {
            Write-Result "PASS" "Usable bash found at $candidate; output: $outputText"
            $usable = $true
            break
        }
        Write-Result "WARN" "Bash candidate failed at $candidate; exit=$exitCode; output=$outputText"
    } catch {
        Write-Result "WARN" "Bash candidate could not be executed at $candidate; error=$($_.Exception.Message)"
    }
}

if (-not $usable) {
    Write-Result "LOCAL_BASH_BLOCKED_BY_WINDOWS_WSL_SETUP" "This Windows machine resolves bash to an unusable WSL path. Option A: install Git Bash and call C:\Program Files\Git\bin\bash.exe. Option B: repair WSL. Option C: use supported PowerShell scripts. Option D: use GitHub Actions Ubuntu evidence for Bash script verification."
}

foreach ($tool in @("docker", "kind", "kubectl")) {
    $command = Get-Command $tool -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        Write-Result "BLOCKED_BY_LOCAL_SETUP" "$tool is not installed or not on PATH."
    } else {
        Write-Result "PASS" "$tool found at $($command.Source)"
    }
}

exit 0
