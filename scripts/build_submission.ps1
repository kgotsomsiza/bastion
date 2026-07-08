# Build (and optionally publish) the Track 1 submission image.
#
# Build + tag only:
#   powershell -ExecutionPolicy Bypass -File scripts/build_submission.ps1 -Registry docker.io/kgotsomsiza
#
# Build, tag, and push publicly (only when ready to submit):
#   powershell -ExecutionPolicy Bypass -File scripts/build_submission.ps1 -Registry docker.io/kgotsomsiza -Push
param(
    [Parameter(Mandatory = $true)][string]$Registry,
    [string]$Name = "bastion",
    [string]$Tag = "track1",
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$image = "$Registry/$Name`:$Tag"

Write-Host "Running tests first..."
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed; not building." }

Write-Host "Building linux/amd64 image $image"
docker buildx build --platform linux/amd64 -t $image -t "$Name`:$Tag" .
if ($LASTEXITCODE -ne 0) { throw "Docker build failed." }

Write-Host "Verifying offline container contract..."
$smokeDir = Join-Path $env:TEMP "bastion-smoke"
New-Item -ItemType Directory -Force -Path "$smokeDir\input", "$smokeDir\output" | Out-Null
Copy-Item data\sample_tasks.json "$smokeDir\input\tasks.json" -Force
docker run --rm -e FRUGAL_ALLOW_REMOTE=0 -v "$smokeDir\input:/input" -v "$smokeDir\output:/output" $image
if ($LASTEXITCODE -ne 0) { throw "Container smoke test failed (exit $LASTEXITCODE)." }
if (-not (Test-Path "$smokeDir\output\results.json")) { throw "Container did not write results.json." }
Write-Host "Contract OK: reads /input/tasks.json, writes /output/results.json, exits 0."

if ($Push) {
    Write-Host "Pushing $image (this makes the image publicly pullable)..."
    docker push $image
    if ($LASTEXITCODE -ne 0) { throw "Push failed." }
    Write-Host "Pushed. Submission image: $image"
} else {
    Write-Host "Not pushing (-Push not set). Image is built and tagged locally as $image"
}
