require 'find'

files = []
Find.find("testfiles") do |fname|
    next unless fname =~ /\.xz$/
    files.push(fname)
end

encoder = "xz -c -6e"
#encoder = "brotli --best -c"
#encoder = "lz4 -9 -c"

if `echo "test" | sha256sum`.split[0] != "f2ca1bb6c7e907d06dafe4687e579fce76b37e4e93b7605022da52e6ccc26fd2"
  abort "sha256sum doesn't work\n"
end

printf("| Test sample                              | normal compression | compression after filter | delta  |\n")
printf("| ---------------------------------------- | ------------------ | ------------------------ | ------ |\n")
files.each do |fname|
  # corectness test
  expected_hash = `xz -kdc #{fname} | sha256sum`
  encoded_hash = `xz -kdc #{fname} | ./xztextp1 | sha256sum`
  decoded_hash = `xz -kdc #{fname} | ./xztextp1 | ./xztextp1 -d | sha256sum`
  unless encoded_hash != decoded_hash && decoded_hash == expected_hash
    abort "Error: reversibility test failed\n"
  end

  # size reduction test
  filtered = `xz -kdc #{fname} | ./xztextp1 | #{encoder}`
  normal = `xz -kdc #{fname} | #{encoder}`
  increase = (filtered.bytesize.to_f / normal.bytesize.to_f) * 100 - 100
  printf("| %-40s | %18d | %24d | %s%.2f%% |\n", File.basename(fname).gsub(/\.xz$/, ""),
         normal.bytesize, filtered.bytesize,
         (increase >= 0 ? "+" : ""), increase)
end
