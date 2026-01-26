# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\scripts\download_netlib_kennington.ps1

$base   = "https://www.netlib.org/lp/data/kennington/"
$index  = $base + "index.html"
$target = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/kennington"

# Path to emps.exe from your netlib folder
$empsExe = "C:/Users/k24095864/C++project/PD-PMM_SSN/data/netlib/emps.exe"
if (-not (Test-Path $empsExe)) {
  throw "emps.exe not found at: $empsExe. Run your netlib downloader first (or fix the path)."
}

New-Item -ItemType Directory -Force -Path $target | Out-Null

Write-Host "Fetching Kennington index..."
$html = Invoke-WebRequest $index

# Grab all .gz links (works whether they are foo.gz or foo.mps.gz)
$files = $html.Links |
  Where-Object { $_.href -and $_.href -match "\.gz$" } |
  Select-Object -ExpandProperty href

Write-Host "Found $($files.Count) .gz files."

function Expand-GzipFile {
  param(
    [Parameter(Mandatory=$true)][string]$InFile,
    [Parameter(Mandatory=$true)][string]$OutFile
  )
  $inStream   = [IO.File]::OpenRead($InFile)
  $gzipStream = New-Object IO.Compression.GzipStream($inStream, [IO.Compression.CompressionMode]::Decompress)
  $outStream  = [IO.File]::Create($OutFile)

  $buffer = New-Object byte[] 8192
  while (($read = $gzipStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
    $outStream.Write($buffer, 0, $read)
  }

  $gzipStream.Dispose()
  $outStream.Dispose()
  $inStream.Dispose()
}

# Run emps in the kennington folder so outputs land there
Push-Location $target

foreach ($f in $files) {
  $url    = $base + $f
  $gzPath = Join-Path $target $f

  Write-Host "Downloading $f ..."
  Invoke-WebRequest -Uri $url -OutFile $gzPath

  # Decompressed raw filename = strip only the final ".gz"
  $rawPath = $gzPath -replace "\.gz$", ""

  Write-Host "Unzipping $f -> $(Split-Path $rawPath -Leaf) ..."
  Expand-GzipFile -InFile $gzPath -OutFile $rawPath
  Remove-Item $gzPath -Force

  # If it gunzips to something ending with .mps, that's often still emps-compressed.
  # Feed the *raw file* into emps.exe; it will create NAME.mps (NAME from inside the file).
  $rawLeaf = Split-Path $rawPath -Leaf

  Write-Host "Expanding via emps.exe: $rawLeaf ..."
  & $empsExe -S $rawLeaf

  if ($LASTEXITCODE -ne 0) {
    Write-Warning "emps.exe failed on $rawLeaf (exit code $LASTEXITCODE). Keeping raw file for inspection: $rawPath"
  } else {
    # On success, remove the raw compressed file; keep only the produced .mps
    Remove-Item $rawPath -Force
  }
}

Pop-Location

Write-Host "Done. Expanded .mps files are in $target"
