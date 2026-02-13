param(
    [ValidateSet("phone", "phone-nospeed", "pc", "pc-nospeed", "both", "all")]
    [string]$Target = "all",
    [switch]$SkipFigures
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

function Invoke-FigureGeneration {
    param(
        [ValidateSet("phone", "desktop")]
        [string]$Profile
    )

    if ($SkipFigures) {
        Write-Host "[build] Skipping figure generation ($Profile)."
        return
    }

    Write-Host "[build] Generating figures for profile: $Profile"
    Push-Location $repoRoot
    try {
        & python silver.py --figure-profile $Profile --figure-root documentation/figures
        if ($LASTEXITCODE -ne 0) {
            throw "Figure generation failed for profile '$Profile' (exit code $LASTEXITCODE)."
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-LatexBuild {
    param(
        [string]$TexFile
    )

    Write-Host "[build] Compiling $TexFile"
    Push-Location $scriptDir
    try {
        & latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -g $TexFile
        if ($LASTEXITCODE -ne 0) {
            throw "LaTeX build failed for '$TexFile' (exit code $LASTEXITCODE)."
        }
    }
    finally {
        Pop-Location
    }
}

switch ($Target) {
    "phone" {
        Invoke-FigureGeneration -Profile "phone"
        Invoke-LatexBuild -TexFile "main-phone.tex"
    }
    "phone-nospeed" {
        Invoke-FigureGeneration -Profile "phone"
        Invoke-LatexBuild -TexFile "main-phone-nospeed.tex"
    }
    "pc" {
        Invoke-FigureGeneration -Profile "desktop"
        Invoke-LatexBuild -TexFile "main-pc.tex"
    }
    "pc-nospeed" {
        Invoke-FigureGeneration -Profile "desktop"
        Invoke-LatexBuild -TexFile "main-pc-nospeed.tex"
    }
    "both" {
        Invoke-FigureGeneration -Profile "phone"
        Invoke-LatexBuild -TexFile "main-phone.tex"
        Invoke-FigureGeneration -Profile "desktop"
        Invoke-LatexBuild -TexFile "main-pc.tex"
    }
    "all" {
        Invoke-FigureGeneration -Profile "phone"
        Invoke-LatexBuild -TexFile "main-phone.tex"
        Invoke-LatexBuild -TexFile "main-phone-nospeed.tex"
        Invoke-FigureGeneration -Profile "desktop"
        Invoke-LatexBuild -TexFile "main-pc.tex"
        Invoke-LatexBuild -TexFile "main-pc-nospeed.tex"
    }
    default {
        throw "Unsupported target: $Target"
    }
}

Write-Host "[build] Done. Output PDFs in $scriptDir"
