# read an UTF-8 text from stdout and analyze its properties

freqblk = {}
freqch = {}

STDIN.read.unpack("U*").each do |ch|
  uniblock = ch.ord >> 7
  freqblk[uniblock] = (freqblk[uniblock] || 0) + 1
  freqch[ch.ord] = (freqch[ch.ord] || 0) + 1
end

freqblk.to_a.sort {|a, b| b[1] <=> a[1] }.take(10).each do |x|
  uniqcnt = 0
  (x[0] << 7).upto((x[0] << 7) + 0x7F).each {|y| uniqcnt += 1 if freqch.has_key?(y) }
  printf("%04X..%04X range: %10d hits (%d out of 128 actually used)\n",
         x[0] << 7, (x[0] << 7) + 0x7F, x[1], uniqcnt)
end
