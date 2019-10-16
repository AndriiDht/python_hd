from numba import uint32
from trsfile import trs_open, Trace, SampleCoding, TracePadding, Header
import struct
import struct
import os

mask32bit = ((1 << 32) - 1)


K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]


def RR(x, shift): return ((x >> shift) | (x << (32 - shift))) & mask32bit


def Sigma0(x): return RR(x, 2) ^ RR(x, 13) ^ RR(x, 22)


def Sigma1(x): return RR(x, 6) ^ RR(x, 11) ^ RR(x, 25)


def Ch(ee, ff, gg): return (ee & ff) ^ (uint32(~ee & mask32bit) & gg)


def Maj(aa, bb, cc): return (aa & bb) ^ (aa & cc) ^ (bb & cc)


def word2bytes(word):
    textins = [struct.pack('>I', w) for w in word]
    textins_bytes = []
    for textin in textins:
        textins_bytes += [i for i in textin]
    return textins_bytes


# sha256 full cycle (64 rounds) for 16 words
def sha256(word, trs_file):
    assert len(word) == 16, "16 words are needed for full cycle of sha-256"
    input_word = [item for item in word]

    h0 = 0x6a09e667
    h1 = 0xbb67ae85
    h2 = 0x3c6ef372
    h3 = 0xa54ff53a
    h4 = 0x510e527f
    h5 = 0x9b05688c
    h6 = 0x1f83d9ab
    h7 = 0x5be0cd19

    a, b, c, d, e, f, g, h = h0, h1, h2, h3, h4, h5, h6, h7

    # extend words to 64 rounds
    for i in range(16, 64):
        s0 = (RR(word[i - 15], 7) ^ RR(word[i - 15], 18) ^ (word[i - 15] >> 3)) & mask32bit
        s1 = (RR(word[i - 2], 17) ^ RR(word[i - 2],  19) ^ (word[i - 2] >> 10)) & mask32bit
        word.append((word[i - 16] + s0 + word[i - 7] + s1) & mask32bit)

    # main cycle
    one_trace = []
    for i in range(64):
        a, b, c, d, e, f, g, h, hamming_distance = do_round(a, b, c, d, e, f, g, h, K[i], word[i])
        one_trace.append(hamming_distance)

    h0 = (h0 + a) & mask32bit
    h1 = (h1 + b) & mask32bit
    h2 = (h2 + c) & mask32bit
    h3 = (h3 + d) & mask32bit
    h4 = (h4 + e) & mask32bit
    h5 = (h5 + f) & mask32bit
    h6 = (h6 + g) & mask32bit
    h7 = (h7 + h) & mask32bit

    trs_file.append(Trace(SampleCoding.INT, one_trace, bytes(word2bytes(input_word) + word2bytes([h0, h1, h2, h3, h4, h5, h6, h7]))))
    return h0, h1, h2, h3, h4, h5, h6, h7,


def do_round(a, b, c, d, e, f, g, h, key, word):
    old_a, old_b, old_c, old_d, old_e, old_f, old_g, old_h = a, b, c, d, e, f, g, h
    tmp = uint32((Sigma1(e) + Ch(e, f, g) + h + key) & mask32bit)
    preA = uint32((Sigma0(a) + Maj(a, b, c) + tmp) & mask32bit)
    preE = uint32((tmp + d) & mask32bit)

    h = g
    g = f
    f = e
    e = (preE + word) & mask32bit
    d = c
    c = b
    b = a
    a = (preA + word) & mask32bit

    hamming_distance = bin(a ^ old_a).count('1')
    hamming_distance += bin(b ^ old_b).count('1')
    hamming_distance += bin(c ^ old_c).count('1')
    hamming_distance += bin(d ^ old_d).count('1')
    hamming_distance += bin(e ^ old_e).count('1')
    hamming_distance += bin(f ^ old_f).count('1')
    hamming_distance += bin(g ^ old_g).count('1')
    hamming_distance += bin(h ^ old_h).count('1')
    return a, b, c, d, e, f, g, h, hamming_distance


def test0():
    word0 = [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
             0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
             0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
             0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]
    h0, h1, h2, h3, h4, h5, h6, h7 = sha256(word0)
    assert (h0, h1, h2, h3, h4, h5, h6, h7) == \
           (0xef0c748d, 0xf4da50a8, 0xd6c43c01, 0x3edc3ce7, 0x6c9d9fa9, 0xa1458ade, 0x56eb86c0, 0xa64492d2)


def test1():
    word1 = [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
             0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
             0x0, 0x0, 0x0, 0x0,
             0x0, 0x0, 0x0, 0x0, ]
    h0, h1, h2, h3, h4, h5, h6, h7 = sha256(word1)
    assert (h0, h1, h2, h3, h4, h5, h6, h7) == \
           (0xc2c9f7b1, 0x39346d8e, 0xf59b77e9, 0x2cd6ce3c, 0x114a35b7, 0x20f95a23, 0xad4a35c8, 0x3bba1e7e)


def do_tests():
    test0()
    test1()


if __name__ == "__main__":
    word1 = [0xB444681D, 0x19EF2A35, 0x5B5E952B, 0x38D65656,
             0xB444681D, 0x19EF2A35, 0x5B5E952B, 0x38D65656,
             0x2B7E1516, 0x28AED2A6, 0xABF71588, 0x09CF4F3C,
             0x2B7E1516, 0x28AED2A6, 0xABF71588, 0x09CF4F3C, ]

    word2 = [0x2CBD735F, 0xB69496E4, 0x5A981F11, 0xD13EDEE3,
             0x2CBD735F, 0xB69496E4, 0x5A981F11, 0xD13EDEE3,
             0x2B7E1516, 0x28AED2A6, 0xABF71588, 0x09CF4F3C,
             0x2B7E1516, 0x28AED2A6, 0xABF71588, 0x09CF4F3C, ]

    # do_tests()
    with trs_open('sha256_hd_samekey.trs', 'w', engine='TrsEngine', padding_mode=TracePadding.AUTO, live_update=True) as trs_file:
        for i in range(1000000):
            if (i % 1000 == 0):
                print(f'{i} traces written!\r')
            h0, h1, h2, h3, h4, h5, h6, h7 = sha256([item for item in word1], trs_file)
            next_input = [h0, h1, h2, h3, h4, h5, h6, h7]


    # print(hex(h0), hex(h1), hex(h2), hex(h3), hex(h4), hex(h5), hex(h6), hex(h7))
