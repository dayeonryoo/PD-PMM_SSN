$base   = "https://www.netlib.org/lp/data/"
$target = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/netlib"

New-Item -ItemType Directory -Force -Path $target | Out-Null

Write-Host "Fetching index from $base ..."
$html = Invoke-WebRequest $base

# Get all compressed LP files (no '/', not queries, and not obvious code/docs)
$files = $html.Links |
    Where-Object {
        $_.href -and
        -not $_.href.StartsWith("?") -and
        -not $_.href.Contains("/")   -and
        ($_.'href' -notmatch '\.c$|\.f$|\.gz$|readme|ascii|nams\.ps')
    } |
    Select-Object -ExpandProperty href

Write-Host "Found $($files.Count) compressed LP files."

# Download all compressed LP files (no extension)
foreach ($f in $files) {
    $url = $base + $f
    $out = Join-Path $target $f
    Write-Host "Downloading $f ..."
    Invoke-WebRequest -Uri $url -OutFile $out
}

Write-Host "Done downloading compressed LP files to $target."

# -------- Ensure emps.exe exists (download + decompress if necessary) --------

$empsExe = Join-Path $target "emps.exe"
$empsGz  = Join-Path $target "emps.exe.gz"

if (-not (Test-Path $empsExe)) {
    Write-Host "emps.exe not found, downloading emps.exe.gz ..."
    Invoke-WebRequest -Uri ($base + "emps.exe.gz") -OutFile $empsGz

    Write-Host "Decompressing emps.exe.gz -> emps.exe ..."

    # Small helper to expand a gzip file
    function Expand-GzipFile {
        param(
            [Parameter(Mandatory=$true)][string]$InFile,
            [Parameter(Mandatory=$true)][string]$OutFile
        )

        $inStream  = [IO.File]::OpenRead($InFile)
        $gzipStream = New-Object IO.Compression.GzipStream($inStream, [IO.Compression.CompressionMode]::Decompress)
        $outStream = [IO.File]::Create($OutFile)

        $buffer = New-Object byte[] 4096
        while (($read = $gzipStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $outStream.Write($buffer, 0, $read)
        }

        $gzipStream.Dispose()
        $outStream.Dispose()
        $inStream.Dispose()
    }

    Expand-GzipFile -InFile $empsGz -OutFile $empsExe
    Remove-Item $empsGz -Force

    Write-Host "emps.exe ready."
} else {
    Write-Host "Found existing emps.exe."
}

# -------- Expand compressed LPs to .mps using emps.exe --------

Set-Location $target
Write-Host "Expanding compressed LP files to .mps ..."

foreach ($f in $files) {
    Write-Host "Expanding $f ..."
    & $empsExe -S $f    # this creates NAME.mps (NAME from inside the file)
}

Write-Host "Expansion complete."

# -------- CLEANUP: delete compressed originals, keep only .mps --------

Write-Host "Deleting compressed source files (non-.mps) ..."

foreach ($f in $files) {
    $path = Join-Path $target $f
    if (Test-Path $path) {
        Remove-Item $path -Force
    }
}

Write-Host "Done. Only .mps files (and emps.exe) remain in $target."
