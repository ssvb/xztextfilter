# Remove "unintresting" sentences with fewer than 10% nonascii
# characters and also remove timestamps

def interesting(str)
  total = 0
  ascii = 0
  str.unpack("U*").each do |ch|
    ascii += 1 unless ch.ord >= 0x80
    total += 1
  end
  return ascii.to_f / total < 0.90
end

out = File.open("tatoeba-nonascii-CC0.csv", "wb")

File.open("sentences_CC0.csv").each_line do |l|
  tmp = l.strip.split("\t")
  next unless interesting(tmp[2])
  out.printf("%s\t%s\t%s\n", tmp[0], tmp[1], tmp[2])
end
