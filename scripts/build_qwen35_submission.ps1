# Build the Qwen3.5 confidence-tier submission from a minimal staged context.
param(
    [Parameter(Mandatory = $true)][string]$Registry,
    [string]$Name = "bastion",
    [string]$Tag = "track1-v23-qwen35",
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$image = "$Registry/$Name`:$Tag"
$modelSource = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\models\Qwen3.5-4B-Q4_K_M.gguf"))
$expectedHash = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
$expectedBytes = 2740937888

if (-not (Test-Path -LiteralPath $modelSource)) {
    throw "Missing model: $modelSource"
}
$model = Get-Item -LiteralPath $modelSource
if ($model.Length -ne $expectedBytes) {
    throw "Model size mismatch: expected $expectedBytes bytes, got $($model.Length)"
}
$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelSource).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) {
    throw "Model checksum mismatch: expected $expectedHash, got $actualHash"
}

$commit = (git rev-parse --short HEAD).Trim()
$fullCommit = (git rev-parse HEAD).Trim()
$dirty = git status --porcelain --untracked-files=no
if ($dirty) {
    throw "Tracked worktree changes exist. Commit them before building an auditable image."
}
$builtAt = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")

$tempRoot = [IO.Path]::GetFullPath($env:TEMP).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$stage = [IO.Path]::GetFullPath((Join-Path $tempRoot "bastion-qwen35-build-$PID"))
if (-not $stage.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Staging path escaped the temporary directory: $stage"
}

try {
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Copy-Item -Recurse -Force (Join-Path $PSScriptRoot "..\frugalrouter") $stage
    Copy-Item -Recurse -Force (Join-Path $PSScriptRoot "..\config") $stage
    Copy-Item -Recurse -Force (Join-Path $PSScriptRoot "..\wheels") $stage
    Copy-Item -Force (Join-Path $PSScriptRoot "..\Dockerfile.qwen35") (Join-Path $stage "Dockerfile")
    New-Item -ItemType Directory -Force -Path (Join-Path $stage "models") | Out-Null
    New-Item -ItemType HardLink -Path (Join-Path $stage "models\model.gguf") -Target $modelSource | Out-Null

    Write-Host "Running tests..."
    python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Tests failed." }

    Write-Host "Building $image for linux/amd64 from commit $commit..."
    docker buildx build --platform linux/amd64 --load `
        --label "bastion.commit=$commit" `
        --label "org.opencontainers.image.revision=$fullCommit" `
        --label "bastion.built=$builtAt" `
        -t $image $stage
    if ($LASTEXITCODE -ne 0) { throw "Docker build failed." }

    $smokeDir = [IO.Path]::GetFullPath((Join-Path $env:TEMP "bastion-qwen35-smoke-$PID"))
    if (-not $smokeDir.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Smoke path escaped the temporary directory: $smokeDir"
    }
    New-Item -ItemType Directory -Force -Path "$smokeDir\input", "$smokeDir\output" | Out-Null
    Copy-Item (Join-Path $PSScriptRoot "..\data\sample_tasks.json") "$smokeDir\input\tasks.json" -Force

    Write-Host "Testing the container contract under 2 CPU / 4 GB..."
    docker run --rm --cpus=2 --memory=4g -e FRUGAL_ALLOW_REMOTE=0 `
        -v "$smokeDir\input:/input" -v "$smokeDir\output:/output" $image
    if ($LASTEXITCODE -ne 0) { throw "Container smoke test failed." }
    if (-not (Test-Path "$smokeDir\output\results.json")) {
        throw "Container did not write results.json."
    }

    if ($Push) {
        Write-Host "Publishing linux/amd64 OCI image $image..."
        docker buildx build --platform linux/amd64 --push `
            --label "bastion.commit=$commit" `
            --label "org.opencontainers.image.revision=$fullCommit" `
            --label "bastion.built=$builtAt" `
            -t $image $stage
        if ($LASTEXITCODE -ne 0) { throw "Docker push failed." }
        docker buildx imagetools inspect $image
        if ($LASTEXITCODE -ne 0) { throw "Published-image inspection failed." }
    }
} finally {
    if ($smokeDir -and (Test-Path -LiteralPath $smokeDir)) {
        $resolvedSmoke = [IO.Path]::GetFullPath($smokeDir)
        if ($resolvedSmoke.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolvedSmoke -Recurse -Force
        }
    }
    if (Test-Path -LiteralPath $stage) {
        $resolvedStage = [IO.Path]::GetFullPath($stage)
        if ($resolvedStage.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolvedStage -Recurse -Force
        }
    }
}
