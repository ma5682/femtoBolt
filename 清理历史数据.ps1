param(
    [ValidateSet('exports','history')][string]$Scope = 'exports',
    [switch]$ListOnly,
    [switch]$Interactive
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'commercial_compliance_check.py'))) {
    throw '请将脚本放在 Femto 项目根目录运行。'
}
if ($Interactive) {
    Write-Host '请先关闭采集、重建及模型预览窗口。'
    Write-Host '1. 仅清理区域 / 路径导出（保留原始采集和模型，默认）'
    Write-Host '2. 清理全部采集历史（原始 RGB/深度/位姿、重建模型、圈选和路径导出）'
    Write-Host '其他诊断证据、代码、环境和模型依赖不会删除。'
    $choice = Read-Host '输入 1 或 2；直接回车选择 1；输入 Q 退出'
    if ($choice -eq 'Q') { return }
    if ($choice -eq '2') { $Scope = 'history' }
    elseif ($choice -eq '1' -or $choice -eq '') { $Scope = 'exports' }
    else { throw '选项无效，未删除任何文件。' }
}
$targets = [Collections.Generic.List[string]]::new()
function Add-CheckedTarget([string]$candidate, [string]$allowedParent) {
    if (-not (Test-Path -LiteralPath $candidate)) { return }
    $resolved = (Resolve-Path -LiteralPath $candidate).ProviderPath
    $allowed = [IO.Path]::GetFullPath($allowedParent).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
        throw "目标越界，已停止：$resolved"
    }
    # Reject junctions/symlinks anywhere in the target or its ancestors.
    $item = Get-Item -LiteralPath $resolved -Force
    $ancestor = $item
    while ($null -ne $ancestor) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "拒绝链接目录：$($ancestor.FullName)" }
        if ($ancestor.FullName.TrimEnd('\') -eq $projectRoot) { break }
        $ancestor = if ($ancestor -is [IO.DirectoryInfo]) { $ancestor.Parent } else { $ancestor.Directory }
    }
    if ($item.PSIsContainer) {
        $links = @(Get-ChildItem -LiteralPath $resolved -Recurse -Force -Attributes ReparsePoint)
        if ($links.Count) { throw "目标含链接，未执行清理：$resolved" }
    }
    if (-not $targets.Contains($resolved)) { $targets.Add($resolved) }
}
$outputs = Join-Path $projectRoot 'outputs'
$datasets = Join-Path $projectRoot 'datasets'
$models = @(Get-ChildItem -LiteralPath $outputs -Directory -Filter 'complete_demo_*')
if ($Scope -eq 'exports') {
    foreach ($model in $models) {
        foreach ($name in @('annotations','region_paths')) {
            $parent = Join-Path $model.FullName "reconstruction\$name"
            if (Test-Path -LiteralPath $parent) {
                foreach ($entry in Get-ChildItem -LiteralPath $parent -Directory) {
                    Add-CheckedTarget $entry.FullName $outputs
                }
            }
        }
    }
} else {
    foreach ($entry in Get-ChildItem -LiteralPath $datasets -Directory -Filter 'scene_*') {
        Add-CheckedTarget $entry.FullName $datasets
    }
    foreach ($model in $models) { Add-CheckedTarget $model.FullName $outputs }
    foreach ($name in @('latest_complete_demo.txt','latest_moving_reconstruction.txt','local_history_deleted.json')) {
        Add-CheckedTarget (Join-Path $outputs $name) $outputs
    }
}
Write-Host "`n清理范围：$Scope；目标数量：$($targets.Count)"
foreach ($target in $targets) { Write-Host $target }
if ($ListOnly -or $targets.Count -eq 0) { return }
Write-Host '`n将移入回收站，不直接永久删除。大量数据可能需要一些时间。'
$confirmation = Read-Host '确认清理请输入 DELETE；其他输入取消'
if ($confirmation -cne 'DELETE') { Write-Host '已取消，未删除文件。'; return }
Add-Type -AssemblyName Microsoft.VisualBasic
$log = [Collections.Generic.List[string]]::new()
$log.Add("Scope=$Scope; Started=$(Get-Date -Format o)")
foreach ($target in $targets) {
    try {
        # Recheck after confirmation to avoid acting on a changed link/target.
        $allowedParent = if ($target.StartsWith($datasets + '\',[StringComparison]::OrdinalIgnoreCase)) { $datasets } else { $outputs }
        Add-CheckedTarget $target $allowedParent
        if (Test-Path -LiteralPath $target -PathType Container) {
            [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($target, 'OnlyErrorDialogs', 'SendToRecycleBin', 'ThrowException')
        } elseif (Test-Path -LiteralPath $target -PathType Leaf) {
            [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($target, 'OnlyErrorDialogs', 'SendToRecycleBin', 'ThrowException')
        }
        $log.Add("RECYCLED $target"); Write-Host "已移入回收站：$target"
    } catch {
        $log.Add("FAILED $target : $($_.Exception.Message)")
        Write-Host "失败，已停止后续清理：$($_.Exception.Message)" -ForegroundColor Red
        break
    }
}
$logPath = Join-Path $projectRoot ('历史清理日志_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.txt')
$log | Set-Content -LiteralPath $logPath -Encoding UTF8
Write-Host "日志：$logPath"
