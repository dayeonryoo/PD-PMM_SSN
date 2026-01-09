$base   = "http://www.netlib.org/lp/data/"
$target = "data/netlib"

New-Item -ItemType Directory -Force -Path $target | Out-Null

Write-Host "Fetching index from $base ..."
$html = Invoke-WebRequest $base

# Get all hrefs that look like files (skip parent dirs / queries / subdirs)
$files = $html.Links |
    Where-Object { $_.href -and -not $_.href.StartsWith("?") -and -not $_.href.Contains("/") } |
    Select-Object -ExpandProperty href

foreach ($f in $files) {
    $url = $base + $f
    $out = Join-Path $target $f
    Write-Host "Downloading $f ..."
    Invoke-WebRequest -Uri $url -OutFile $out
}

Write-Host "Done. Files saved in $target"
