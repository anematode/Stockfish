#include <utility>

template <IndexType Dimensions>
struct NNZInfo {
    //uint8_t bitset[(Dimensions + 31) / 32];

    using NNZIndex = std::conditional_t<Dimensions > 1024, uint16_t, uint8_t>;
    NNZIndex nnz[Dimensions / 4];
    size_t count;

    struct NNZCursor {
        // uint8_t* out;

        NNZIndex* out;
        size_t* count;

        __m512i indices;
        __m512i increment;

        NNZCursor(NNZInfo& info, IndexType offset) {

        }


        void record2(vec_t neurons1, vec_t neurons2) {
    #if defined(USE_AVX512ICL)

        constexpr IndexType SimdWidthIn  = 64;  // 512 bits
        constexpr IndexType SimdWidthOut = 32;  // 512 bits / 16 bits
        constexpr IndexType SimdChunks   = NumChunks / SimdWidthOut;
        const __m512i       increment    = _mm512_set1_epi16(SimdWidthOut);
        __m512i             base = _mm512_set_epi16(  // Same permute order as _mm512_packus_epi32()
          31, 30, 29, 28, 15, 14, 13, 12, 27, 26, 25, 24, 11, 10, 9, 8, 23, 22, 21, 20, 7, 6, 5, 4,
          19, 18, 17, 16, 3, 2, 1, 0);

        IndexType count = 0;
        for (IndexType i = 0; i < SimdChunks; ++i)
        {
            // Get a bitmask and gather non zero indices
            const __m512i   inputV01 = _mm512_packs_epi32(neurons1, neurons2);
            const __mmask32 nnzMask  = _mm512_test_epi16_mask(inputV01, inputV01);

            // Avoid _mm512_mask_compressstoreu_epi16() as it's 256 uOps on Zen4
            __m512i nnz = _mm512_maskz_compress_epi16(nnzMask, base);
            _mm512_storeu_si512(out + count, nnz);

            count += popcount(nnzMask);
            base = _mm512_add_epi16(base, increment);
        }
        count_out = count;

    #elif defined(USE_AVX512)

        constexpr IndexType SimdWidth  = 16;  // 512 bits / 32 bits
        constexpr IndexType SimdChunks = NumChunks / SimdWidth;
        const __m512i       increment  = _mm512_set1_epi32(SimdWidth);
        __m512i base = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        IndexType count = 0;
        for (IndexType i = 0; i < SimdChunks; ++i)
        {
            const __m512i inputV = _mm512_load_si512(input + i * SimdWidth * sizeof(std::uint32_t));

            // Get a bitmask and gather non zero indices
            const __mmask16 nnzMask = _mm512_test_epi32_mask(inputV, inputV);
            const __m512i   nnzV    = _mm512_maskz_compress_epi32(nnzMask, base);
            _mm512_mask_cvtepi32_storeu_epi16(out + count, 0xFFFF, nnzV);
            count += popcount(nnzMask);
            base = _mm512_add_epi32(base, increment);
        }
        count_out = count;

        }

private:
        void record(vec_t neurons) {

        }
    };

    NNZCursor make_cursor(IndexType offset) const {
        r
    }
};
