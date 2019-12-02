$root_dir = "path/to/root/dir"
$hash_file = "path/to/hash/file"

Get-ChildItem -File -Recurse | % {
  $path = (Resolve-Path $_.FullName -Relative)
  $result = certutil -hashfile "$path" SHA256
  echo ($result[0] + "," + $result[1] + "," + $result[2])
} | Tee-Object -FilePath $hash_file

