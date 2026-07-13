# Demonstrate the exact public V23 image with route and token metadata.
param(
    [string]$Image = "docker.io/kgotsomsiza/bastion:track1-v23",
    [string]$Prompt = "What is the chemical symbol for gold? Answer with only the symbol.",
    [switch]$AllowRemote
)

$ErrorActionPreference = "Stop"
$tempBase = [IO.Path]::GetFullPath($env:TEMP).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$runDir = [IO.Path]::GetFullPath((Join-Path $tempBase "bastion-v23-demo-$PID"))
if (-not $runDir.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Demo path escaped the temporary directory: $runDir"
}

try {
    $inputDir = Join-Path $runDir "input"
    $outputDir = Join-Path $runDir "output"
    $logsDir = Join-Path $runDir "logs"
    New-Item -ItemType Directory -Force -Path $inputDir, $outputDir, $logsDir | Out-Null

    $taskJson = @(@{ task_id = "v23-demo"; prompt = $Prompt }) | ConvertTo-Json -Compress -Depth 5
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText((Join-Path $inputDir "tasks.json"), $taskJson, $utf8NoBom)

    $dockerArgs = @(
        "run", "--rm", "--cpus=2", "--memory=4g",
        "-e", "FRUGAL_WORKERS=1",
        "-e", "FRUGAL_DECISION_LOG=/logs/decisions.jsonl",
        "-v", "${inputDir}:/input",
        "-v", "${outputDir}:/output",
        "-v", "${logsDir}:/logs"
    )

    if ($AllowRemote) {
        $key = [Environment]::GetEnvironmentVariable("FIREWORKS_API_KEY", "User")
        if (-not $key) { throw "FIREWORKS_API_KEY is not set in the user environment." }
        $dockerArgs += @(
            "-e", "FIREWORKS_API_KEY=$key",
            "-e", "FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1",
            "-e", "ALLOWED_MODELS=accounts/fireworks/models/gemma-4-31b-it,accounts/fireworks/models/kimi-k2p7-code"
        )
    }
    else {
        $dockerArgs += @("-e", "FRUGAL_ALLOW_REMOTE=0")
    }

    $dockerArgs += @(
        $Image,
        "python", "-m", "frugalrouter.cli",
        "--tasks", "/input/tasks.json",
        "--output", "/output/results.jsonl",
        "--debug-output",
        "--workers", "1",
        "--decision-log", "/logs/decisions.jsonl"
    )
    if (-not $AllowRemote) { $dockerArgs += "--no-remote" }

    Write-Host "Running the public V23 image under 2 CPU / 4 GB..." -ForegroundColor Yellow
    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) { throw "V23 container exited with code $LASTEXITCODE." }

    $row = Get-Content -LiteralPath (Join-Path $outputDir "results.jsonl") |
        Where-Object { $_.Trim() } |
        Select-Object -First 1 |
        ConvertFrom-Json
    $tokens = [int]$row.usage.prompt_tokens + [int]$row.usage.completion_tokens

    Write-Host ""
    Write-Host "PROMPT    : $Prompt"
    Write-Host "-----------------------------------------------------------"
    Write-Host "ANSWER    : $($row.answer)" -ForegroundColor Green
    Write-Host "ROUTE     : $($row.route)   (category: $($row.category))"
    Write-Host "MODEL     : $($row.usage.model)"
    Write-Host "TOKENS    : $tokens Fireworks tokens"
    Write-Host "LATENCY   : $($row.usage.latency_ms) ms"
    if ($row.fallback_reason) { Write-Host "NOTE      : $($row.fallback_reason)" }
    Write-Host ""

    if (-not $AllowRemote -and $row.route -ne "local_model" -and $row.route -ne "local") {
        throw "This prompt did not clear V23's local safety gate. Try another validated factual prompt."
    }
}
finally {
    if (Test-Path -LiteralPath $runDir) {
        $resolved = [IO.Path]::GetFullPath($runDir)
        if ($resolved.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolved -Recurse -Force
        }
    }
}
