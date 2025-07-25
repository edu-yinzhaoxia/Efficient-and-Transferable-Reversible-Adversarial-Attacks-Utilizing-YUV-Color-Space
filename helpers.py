
def getfreqs(stream):
    """
    Count the frequency of each symbol in the stream.
    Args:
        stream (list): Input data stream.
    Returns:
        dict: Frequency dictionary of symbols.
    """
    symbols = set(stream)
    freq_dict = {}
    for s in symbols:
        freq_dict[s] = stream.count(s)
    return freq_dict

def Cumfreq(symbol, dictionary):
    """
    Calculate the cumulative frequency up to the given symbol.
    Args:
        symbol: The symbol to accumulate to.
        dictionary (dict): Frequency dictionary.
    Returns:
        int: Cumulative frequency.
    """
    cum = 0
    for sym in dictionary:
        cum += dictionary[sym]
        if sym == symbol:
            break
    return cum

def Arithmetic_encode(stream, precision=32):
    """
    Arithmetic encoding for the input stream.
    Args:
        stream (list): Input stream to encode.
        precision (int): Number of bits for encoding precision.
    Returns:
        list: Encoded bitstream.
        dict: Symbol frequency dictionary.
    """
    stream.append('!')  # End-of-stream symbol
    StreamSize = len(stream)
    dic = getfreqs(stream)

    full = 2 ** precision
    half = full // 2
    quarter = half // 2

    L = 0
    H = full
    trails = 0
    code = []

    for symbol in stream:
        freqSym = dic[symbol]
        S_high = Cumfreq(symbol, dic)
        S_low = S_high - freqSym
        Range = H - L
        H = L + Range * S_high // StreamSize
        L = L + Range * S_low // StreamSize

        while True:
            if H < half:
                code.extend([0])
                code.extend([1] * trails)
                trails = 0
                L *= 2
                H *= 2
            elif L >= half:
                code.extend([1])
                code.extend([0] * trails)
                trails = 0
                L = 2 * (L - half)
                H = 2 * (H - half)
            elif L >= quarter and H < 3 * quarter:
                trails += 1
                L = 2 * (L - quarter)
                H = 2 * (H - quarter)
            else:
                break
    trails += 1
    if L <= quarter:
        code.extend([0])
        code.extend([1] * trails)
    else:
        code.extend([1])
        code.extend([0] * trails)
    return code, dic

def Arithmetic_decode(code, dic, precision=32):
    """
    Arithmetic decoding for the input bitstream.
    Args:
        code (list): Encoded bitstream.
        dic (dict): Symbol frequency dictionary.
        precision (int): Number of bits for decoding precision.
    Returns:
        list: Decoded message (original stream).
    """
    code_size = len(code)
    stream_size = sum(dic.values())

    full = 2 ** precision
    half = full // 2
    quarter = half // 2

    L = 0
    H = full

    val = 0
    indx = 1
    message = []

    while indx <= precision and indx <= code_size:
        if code[indx - 1] == 1:
            val += 2 ** (precision - indx)
        indx += 1
    flag = 1

    while flag:
        for symbol in dic:
            freqSym = dic[symbol]
            S_high = Cumfreq(symbol, dic)
            S_low = S_high - freqSym
            Range = H - L
            H0 = L + Range * S_high // stream_size
            L0 = L + Range * S_low // stream_size

            if L0 <= val < H0:
                message.append(symbol)
                L = L0
                H = H0
                if symbol == '!':
                    flag = 0
                break

        while True:
            if H < half:
                L *= 2
                H *= 2
                val *= 2
                if indx <= code_size:
                    val += code[indx - 1]
                    indx += 1
            elif L >= half:
                L = 2 * (L - half)
                H = 2 * (H - half)
                val = 2 * (val - half)
                if indx <= code_size:
                    val += code[indx - 1]
                    indx += 1
            elif L >= quarter and H < 3 * quarter:
                L = 2 * (L - quarter)
                H = 2 * (H - quarter)
                val = 2 * (val - quarter)
                if indx <= code_size:
                    val += code[indx - 1]
                    indx += 1
            else:
                break
    message.pop()  # Remove end-of-stream symbol
    return message
