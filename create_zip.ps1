$sourceDir = "c:\Users\lefpa\Downloads\OmniScan3D"
$zipPath = "c:\Users\lefpa\Downloads\OmniScan3D_Release.zip"

if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

# Exclude temporary sqlite databases and pycache to keep zip clean and lightweight
Get-ChildItem -Path $sourceDir -Exclude @("*.pyc", "*.db", "*.db-journal") -Recurse | Where-Object {
    $_.FullName -notmatch "__pycache__" -and $_.FullName -notmatch "\.git"
} | Compress-Archive -DestinationPath $zipPath -Update

Write-Host "Created OmniScan3D_Release.zip successfully at: $zipPath"
