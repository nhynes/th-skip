require 'paths'

readWord = (fw2v) ->
  chars = {}
  while true
    char = fw2v\readChar!
    if char == 0 or char == 32
      break
    elseif char ~= 10
      chars[#chars+1] = char

  return torch.CharStorage(chars)\string!

(w2vBin) ->
  cachePath = paths.thisfile '../data/'..paths.basename(w2vBin, '.bin')..'.t7'
  if paths.filep cachePath
    return table.unpack(torch.load cachePath)

  fw2v = torch.DiskFile(w2vBin)
  nwords, embDim = fw2v\readInt!, fw2v\readInt!

  i2w = {}
  wvecs = torch.FloatTensor(nwords, embDim)\zero!

  fw2v\binary!
  for i=1,nwords
    i2w[i] = readWord fw2v
    wvecs[i] = torch.FloatTensor(fw2v\readFloat embDim)

  wvecs\cdiv wvecs\norm(2, 2)\expandAs(wvecs)

  torch.save cachePath, {wvecs, i2w}

  wvecs, i2w
