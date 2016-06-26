require 'paths'
_ = require 'moses'

DataLoader = torch.class('DataLoader')

UNK, SOR, EOR, SOS, EOS = 1, 2, 3, 4, 5

groupByLen = (data) ->
  slens = data.slens

  indsByLen = {}
  for i=1,slens\size(1)
    slen = slens[i]

    ibrl = indsByLen[slen] or {}
    ibrl[#ibrl+1] = i
    indsByLen[slen] = ibrl

  lengths = _.keys indsByLen
  table.sort lengths

  lenFreqs = torch.zeros(#lengths)
  lenFreqs[i] = #indsByLen[lengths[i]] for i=1,#lengths -- freq of each index -> len

  data.lengths = lengths
  data.indsByLen = indsByLen
  data.lenFreqs = lenFreqs
  data.isBoundary = {data.rbps[i], true for i=1,data.rbps\size(1)}

DataLoader.__init = (dataTrain, dataVal, opts) =>
  @batchSize = opts.batchSize
  @dataTrain = dataTrain
  @dataVal = dataVal

  @vocabSize = opts.vocabSize

  groupByLen(dataTrain)
  groupByLen(dataVal)

DataLoader.makebatch = (partition='train') =>
  data = partition == 'val' and @dataVal or @dataTrain

  [==[
    1. pick a sentence length
    <until full minibatch>
    2. grab a sentence, add </s>
    3. grab prev and next sentences
      if it's on a recipe boundary (check rbp), prev sentence is <r>
      if the next sentence is on a boundary, next sentence is </r>
    4. add <s> and </s>
  ]==]

  sentlen = data.lengths[torch.multinomial(data.lenFreqs, 1)[1]]
  sentlenInds = data.indsByLen[sentlen]
  batchSize = math.min(#sentlenInds, @batchSize)

  selIndInds = torch.randperm(#sentlenInds)\sub(1, batchSize)

  maxSentLen = data.lengths[#data.lengths] + 2 -- +2 for <s> and </s>

  -- batchIds = torch.CharTensor(batchSize, 11) -- bin search rbps for selInd
  batchSents = torch.LongTensor(batchSize, sentlen+1)\zero! -- +1 for </s>
  batchPrevSents = torch.LongTensor(batchSize, maxSentLen)\zero!
  batchNextSents = torch.LongTensor(batchSize, maxSentLen)\zero!

  toks = data.toks
  nsents = toks\size(1)
  isBoundary = data.isBoundary
  slens = data.slens

  -- length of sentences including trailing </s> but not leading </s>
  maxPrev = 0
  maxNext = 0

  for i=1,batchSize
    selInd = sentlenInds[selIndInds[i]]

    batchSents[i]\sub(1, -2)\copy(toks[selInd]\sub(1, sentlen))
    batchSents\select(2, sentlen+1)\fill(EOS)

    prevSent = batchPrevSents\select(1, i)\sub(2, -1)
    nextSent = batchNextSents\select(1, i)\sub(2, -1)

    if isBoundary[selInd] or selInd == 1
      prevSent[1] = SOR
      prevSent[2] = EOS
      maxPrev = math.max(maxPrev, 1)
    else
      slen = slens[selInd-1]
      prevSent\sub(1, -2)\copy(toks[selInd-1])
      prevSent[slen+1] = EOS
      maxPrev = math.max(maxPrev, slen)

    if isBoundary[selInd+1] or selInd == nsents
      nextSent[1] = EOR
      nextSent[2] = EOS
      maxNext = math.max(maxNext, 1)
    else
      slen = slens[selInd+1]
      nextSent\sub(1, -2)\copy(toks[selInd+1])
      nextSent[slen+1] = EOS
      maxNext = math.max(maxNext, slen)

  batchPrevSents = batchPrevSents\narrow(2, 1, maxPrev+2)
  batchPrevSents\select(2, 1)\fill(SOS)
  batchNextSents = batchNextSents\narrow(2, 1, maxNext+2)
  batchNextSents\select(2, 1)\fill(SOS)

  collectgarbage!

  batchSents, batchPrevSents, batchNextSents

DataLoader.partitionSize = (partition='train') =>
  ds = partition == 'train' and @dataTrain or @dataVal
  ds.toks\size(1)

DataLoader.terminate = =>
