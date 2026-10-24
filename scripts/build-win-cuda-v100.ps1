param(
    [string]$OutputDir = "release-packages",
    [string]$PackageName = "build-win-cuda-v100",
    [string]$BuildName = "build-win-cuda-v100",
    [string]$Target = "",
    [int]$Parallel = 16,
    [switch]$Package = $false,
    [switch]$StdQuantIteration = $false,
    [switch]$AllTests = $false,
    [switch]$SkipStage = $false,
    [switch]$ConfigureOnly = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:MSBUILDDISABLENODEREUSE = "1"

if (Test-Path "C:\qwen-cpp") {
    $repoRoot = "C:\qwen-cpp"
} else {
    $repoRoot = ($PSScriptRoot | Split-Path -Parent)
}
Set-Location $repoRoot

$cudaVer = "12.9"
$cudaArch = "70" # Tesla V100 / Volta sm_70
$cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$cudaPath = Join-Path $cudaBase "v$cudaVer"
$ninjaExe = Get-Command ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
if (-not $ninjaExe) {
    $ninjaExe = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
}

# Ensure ninja is in PATH.
if (Test-Path $ninjaExe) {
    $env:PATH = "$(Split-Path -Parent $ninjaExe);$env:PATH"
} else {
    Write-Host "[FAIL] ninja.exe not found at $ninjaExe"
    exit 1
}

$vsInstaller = "C:\Program Files (x86)\Microsoft Visual Studio\Installer"
if (Test-Path (Join-Path $vsInstaller "vswhere.exe")) {
    $env:PATH = "$vsInstaller;$env:PATH"
}

# Activate MSVC environment for Ninja (cl.exe, link.exe, etc.).
$vcvarsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if (Test-Path $vcvarsPath) {
    Write-Host "[ENV] Activating MSVC via vcvarsall.bat x64"
    cmd /c "`"$vcvarsPath`" x64 > nul && set" | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
} else {
    Write-Host "[WARN] vcvarsall.bat not found - MSVC may not be available"
}

if (-not (Test-Path $cudaPath)) {
    Write-Host "[FAIL] CUDA $cudaVer not found at $cudaPath"
    exit 1
}

$env:CUDA_PATH = $cudaPath
$env:PATH = "$cudaPath\bin;$env:PATH"

$buildDir = Join-Path $repoRoot $BuildName
$pkgDir = Join-Path $repoRoot "$OutputDir\$PackageName"
$binDir = Join-Path $buildDir "bin"

$commonFlags = @(
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON",
    "-DGGML_CCACHE=OFF",
    "-DGGML_CUDA_FA_ALL_QUANTS=OFF",
    "-DGGML_CUDA_KVARN=ON",
    "-DGGML_CUDA_CUB_3DOT2=ON",
    "-DGGML_NATIVE=ON",
    "-DGGML_BACKEND_DL=OFF",
    "-DGGML_RPC=OFF",
    "-DLLAMA_BUILD_BORINGSSL=OFF",
    "-DLLAMA_BUILD_EXAMPLES=ON",
    "-DLLAMA_BUILD_TESTS=$(if ($AllTests) { 'ON' } else { 'OFF' })",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_TOOLS=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=$cudaArch"
)

Write-Host "========================================"
Write-Host "BeeLlama.cpp Windows CUDA 12.9 sm_70 Build (V100)"
Write-Host "CUDA:    $cudaVer"
Write-Host "Arch:    sm_$cudaArch"
Write-Host "Repo:    $repoRoot"
Write-Host "Build:   $buildDir"
Write-Host "Jobs:    $Parallel"
Write-Host "========================================"

# --- CMake Configure ---
Write-Host "`n[CONFIGURE] cmake -S $repoRoot -B $buildDir"
$cmakeArgs = @("-S", $repoRoot, "-B", $buildDir)
$cmakeArgs += $commonFlags
$cmakeArgs += "-DCUDAToolkit_ROOT=$cudaPath"
$cmakeArgs += "-DCMAKE_CUDA_COMPILER=$cudaPath\bin\nvcc.exe"
$cmakeArgs += "-DCMAKE_MAKE_PROGRAM=$ninjaExe"

$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& cmake @cmakeArgs 2>&1 | ForEach-Object { Write-Host $_ }
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] CMake configure failed (exit code $LASTEXITCODE)"
    exit 1
}
Write-Host "[OK] CMake configured successfully for sm_70"

if ($ConfigureOnly) {
    Write-Host "[DONE] ConfigureOnly requested; build skipped"
    exit 0
}

# --- Build ---
$buildArgs = @("--build", $buildDir, "--parallel", "$Parallel")
if ($Target) {
    $buildArgs += @("--target", $Target)
}

Write-Host "`n[BUILD] cmake $($buildArgs -join ' ')"
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& cmake @buildArgs 2>&1 | ForEach-Object { Write-Host $_ }
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Build failed (exit code $LASTEXITCODE)"
    exit 1
}

Write-Host "[OK] Build completed successfully"
