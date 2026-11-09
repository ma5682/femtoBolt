# Run manually in PowerShell. Permanently deletes only archived RESEARCH_ trials.
# Original datasets, complete_demo outputs, application history and source stay.
# Small JSON records and text/scripts are copied to a separate evidence folder.
$ErrorActionPreference = 'Stop'
$cleanupProjectRoot = [System.IO.Directory]::GetParent($PSScriptRoot).FullName
$cleanupOutputsRoot = (Resolve-Path -LiteralPath (Join-Path $cleanupProjectRoot 'outputs')).Path
$cleanupResearchRoot = (Resolve-Path -LiteralPath (Join-Path $cleanupOutputsRoot 'route_order_patches_20261029')).Path
if (-not $cleanupResearchRoot.StartsWith($cleanupOutputsRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Research directory resolved outside project outputs.'
}
if ((Get-Item -LiteralPath $cleanupResearchRoot).Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
    throw 'Research directory is a link. Nothing deleted.'
}
$cleanupTargets = @(Get-ChildItem -LiteralPath $cleanupResearchRoot -Directory -Force |
    Where-Object { $_.Name.StartsWith('RESEARCH_', [System.StringComparison]::Ordinal) })
$cleanupEvidence = Join-Path $cleanupOutputsRoot ('cleanup_evidence_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $cleanupEvidence -ErrorAction Stop | Out-Null
$cleanupCompleted = @()
foreach ($cleanupTarget in $cleanupTargets) {
    $cleanupResolved = (Resolve-Path -LiteralPath $cleanupTarget.FullName).Path
    if (-not $cleanupResolved.StartsWith($cleanupResearchRoot + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
        [System.IO.Directory]::GetParent($cleanupResolved).FullName -ne $cleanupResearchRoot -or
        -not [System.IO.Path]::GetFileName($cleanupResolved).StartsWith('RESEARCH_', [System.StringComparison]::Ordinal)) {
        throw 'Unexpected deletion target.'
    }
    if ($cleanupTarget.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        throw 'Refusing linked deletion target.'
    }
    $cleanupContents = @(Get-ChildItem -LiteralPath $cleanupResolved -Recurse -Force)
    if ($cleanupContents | Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReparsePoint }) {
        throw 'Refusing a target containing linked files or directories.'
    }
    $cleanupFiles = @($cleanupContents | Where-Object { -not $_.PSIsContainer })
    $cleanupRetained = @($cleanupFiles | Where-Object {
        $_.Extension.ToLowerInvariant() -in @('.py', '.md', '.txt', '.log') -or
        ($_.Extension.ToLowerInvariant() -eq '.json' -and $_.Length -le 1048576)
    })
    foreach ($cleanupFile in $cleanupRetained) {
        $cleanupRelative = $cleanupFile.FullName.Substring($cleanupResearchRoot.Length + 1)
        $cleanupDestination = Join-Path $cleanupEvidence $cleanupRelative
        New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($cleanupDestination)) -Force | Out-Null
        Copy-Item -LiteralPath $cleanupFile.FullName -Destination $cleanupDestination
        if ((Get-FileHash -LiteralPath $cleanupFile.FullName -Algorithm SHA256).Hash -ne
            (Get-FileHash -LiteralPath $cleanupDestination -Algorithm SHA256).Hash) {
            throw 'Evidence copy mismatch; current target has not been deleted.'
        }
    }
    $cleanupRecord = [pscustomobject]@{
        path = $cleanupResolved
        files = $cleanupFiles.Count
        bytes = ($cleanupFiles | Measure-Object Length -Sum).Sum
        retained_bytes = ($cleanupRetained | Measure-Object Length -Sum).Sum
    }
    $cleanupRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $cleanupEvidence 'pending.json') -Encoding utf8
    Remove-Item -LiteralPath $cleanupResolved -Recurse -Force
    $cleanupCompleted += $cleanupRecord
    $cleanupCompleted | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $cleanupEvidence 'deleted.json') -Encoding utf8
    Write-Host ('Deleted ' + $cleanupTarget.Name)
}
Write-Host ('Evidence kept at ' + $cleanupEvidence)
Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($cleanupProjectRoot).Substring(0, 1)) |
    Select-Object Name, Used, Free
