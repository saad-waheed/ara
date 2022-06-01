.section .data,"aw",@progbits
.global NFFT
.balign 8
NFFT:
    .word 0x00000004
    .word 0x00000000
.global samples
.balign NR_LANES*4
samples:
    .word 0xc5d9a53a
    .word 0xc64af149
    .word 0xc6a60d3a
    .word 0xc5e92333
    .word 0xc63e666f
    .word 0x465c0acd
    .word 0x467045f0
    .word 0xc6930b8b
.global twiddle
.balign NR_LANES*4
twiddle:
    .word 0x3f800000
    .word 0x00000000
    .word 0x248d3132
    .word 0x3f800000
    .word 0xbf800000
    .word 0x250d3132
    .word 0xa553c9ca
    .word 0xbf800000
.global twiddle_vec
.balign NR_LANES*4
twiddle_vec:
    .word 0x3f800000
    .word 0x00000000
    .word 0x248d3132
    .word 0x3f800000
    .word 0x3f800000
    .word 0x00000000
    .word 0x3f800000
    .word 0x00000000
.global gold_out
.balign NR_LANES*4
gold_out:
    .word 0xc6c386c8
    .word 0xc6c4c796
    .word 0x46818ca7
    .word 0x4615644e
    .word 0xc64f6488
    .word 0x46d5e11a
    .word 0xc5bfe355
    .word 0xc778d71e
