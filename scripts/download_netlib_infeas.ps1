# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\scripts\download_netlib_infeas_all.ps1

$base   = "https://www.netlib.org/lp/infeas/"
$index  = $base + "index.html"
$target = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/netlib_infeas"

# Path to emps.exe
$empsExe = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/netlib/emps.exe"
if (-not (Test-Path $empsExe)) {
  throw "emps.exe not found at: $empsExe. Run your netlib downloader first (or fix the path)."
}

New-Item -ItemType Directory -Force -Path $target | Out-Null

function Expand-GzipFile {
  param(
    [Parameter(Mandatory=$true)][string]$InFile,
    [Parameter(Mandatory=$true)][string]$OutFile
  )

  $inStream   = [IO.File]::OpenRead($InFile)
  $gzipStream = New-Object IO.Compression.GzipStream($inStream, [IO.Compression.CompressionMode]::Decompress)
  $outStream  = [IO.File]::Create($OutFile)

  try {
    $buffer = New-Object byte[] 8192
    while (($read = $gzipStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
      $outStream.Write($buffer, 0, $read)
    }
  }
  finally {
    $gzipStream.Dispose()
    $outStream.Dispose()
    $inStream.Dispose()
  }
}

Write-Host "Fetching index..."
$html = Invoke-WebRequest $index

# Keep only local file links from this directory, excluding parent dirs, anchors, etc.
$files = $html.Links |
  Where-Object {
    $_.href -and
    $_.href -notmatch '^\?' -and
    $_.href -notmatch '^#' -and
    $_.href -notmatch '/$' -and
    $_.href -ne '../' -and
    $_.href -ne './' -and
    $_.href -notmatch '^https?://'
  } |
  Select-Object -ExpandProperty href -Unique

Write-Host "Found $($files.Count) files."

Push-Location $target

foreach ($f in $files) {
  $url = $base + $f
  $downloadPath = Join-Path $target $f

  Write-Host "Downloading $f ..."
  try {
    Invoke-WebRequest -Uri $url -OutFile $downloadPath
  }
  catch {
    Write-Warning "Failed to download $f : $_"
    continue
  }

  # If gzip, decompress
  if ($downloadPath -match '\.gz$') {
    $rawPath = $downloadPath -replace '\.gz$', ''

    Write-Host "Unzipping $(Split-Path $downloadPath -Leaf) -> $(Split-Path $rawPath -Leaf) ..."
    try {
      Expand-GzipFile -InFile $downloadPath -OutFile $rawPath
      Remove-Item $downloadPath -Force
    }
    catch {
      Write-Warning "Failed to unzip $downloadPath : $_"
      continue
    }

    $fileToProcess = $rawPath
  }
  else {
    $fileToProcess = $downloadPath
  }

  # Optional: run emps.exe on every downloaded/unzipped file
  $leaf = Split-Path $fileToProcess -Leaf
  Write-Host "Expanding via emps.exe: $leaf ..."

  & $empsExe -S $leaf

  if ($LASTEXITCODE -ne 0) {
    Write-Warning "emps.exe failed on $leaf (exit code $LASTEXITCODE). Keeping file for inspection: $fileToProcess"
  }
  else {
    # remove the intermediate compressed/raw file and keep the produced .mps
    Remove-Item $fileToProcess -Force
  }
}

Pop-Location

Write-Host "Done. Output files are in $target"