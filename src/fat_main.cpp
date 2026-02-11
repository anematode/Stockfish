/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Fat binary entry point for x86-64 Linux.
// Detects CPU features via cpuid and dispatches to the best available
// architecture-specific Stockfish build linked into this binary.

#include <cpuid.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Architecture entry points -- defined by the per-arch builds linked in.
// These are the original main() functions, renamed via objcopy.
// The make_fat.py script determines which are available.

// Forward declarations for all possible arch entry points.
// Only the ones actually linked in will resolve; the script ensures
// only declared functions that exist are referenced.

// FAT_ENTRY_DECLARATIONS_BEGIN
// (make_fat.py will verify these match the linked architectures)
extern "C" int sf_main_x86_64(int argc, char* argv[]);
extern "C" int sf_main_x86_64_sse3_popcnt(int argc, char* argv[]);
extern "C" int sf_main_x86_64_ssse3(int argc, char* argv[]);
extern "C" int sf_main_x86_64_sse41_popcnt(int argc, char* argv[]);
extern "C" int sf_main_x86_64_avx2(int argc, char* argv[]);
extern "C" int sf_main_x86_64_bmi2(int argc, char* argv[]);
extern "C" int sf_main_x86_64_avxvnni(int argc, char* argv[]);
extern "C" int sf_main_x86_64_avx512(int argc, char* argv[]);
extern "C" int sf_main_x86_64_vnni512(int argc, char* argv[]);
extern "C" int sf_main_x86_64_avx512icl(int argc, char* argv[]);
// FAT_ENTRY_DECLARATIONS_END

struct CpuFeatures {
    bool sse2     = false;
    bool sse3     = false;
    bool ssse3    = false;
    bool sse41    = false;
    bool popcnt   = false;
    bool avx      = false;
    bool avx2     = false;
    bool bmi1     = false;
    bool bmi2     = false;
    bool avxvnni  = false;
    bool avx512f  = false;
    bool avx512bw = false;
    bool avx512dq = false;
    bool avx512vl = false;
    bool avx512vnni  = false;
    bool avx512vbmi  = false;
    bool avx512vbmi2 = false;
    bool avx512bitalg   = false;
    bool avx512vpopcntdq = false;
    bool avx512ifma = false;
    bool is_amd     = false;
    bool is_zen1_2  = false;  // AMD Zen 1 or Zen 2 (slow pdep/pext)
};

static bool os_supports_avx() {
    // Check OSXSAVE bit in ECX from CPUID leaf 1
    uint32_t eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    if (!(ecx & (1 << 27)))  // OSXSAVE
        return false;

    // Check XCR0 for SSE and AVX state saving support
    uint32_t xcr0;
    __asm__ volatile("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
    return (xcr0 & 0x6) == 0x6;  // SSE state (bit 1) + AVX state (bit 2)
}

static bool os_supports_avx512() {
    if (!os_supports_avx())
        return false;

    uint32_t xcr0;
    __asm__ volatile("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
    // AVX-512 requires opmask (bit 5), ZMM hi256 (bit 6), Hi16_ZMM (bit 7)
    return (xcr0 & 0xe6) == 0xe6;
}

static CpuFeatures detect_cpu() {
    CpuFeatures f;

    uint32_t eax, ebx, ecx, edx;
    uint32_t max_leaf;

    // Leaf 0: vendor string and max leaf
    __cpuid(0, max_leaf, ebx, ecx, edx);

    // Check for AMD (AuthenticAMD)
    if (ebx == 0x68747541 && edx == 0x69746e65 && ecx == 0x444d4163)
        f.is_amd = true;

    if (max_leaf < 1)
        return f;

    // Leaf 1: basic feature flags
    __cpuid(1, eax, ebx, ecx, edx);

    f.sse2   = (edx >> 26) & 1;
    f.sse3   = (ecx >> 0) & 1;
    f.ssse3  = (ecx >> 9) & 1;
    f.sse41  = (ecx >> 19) & 1;
    f.popcnt = (ecx >> 23) & 1;
    f.avx    = (ecx >> 28) & 1;

    // Check OS support for AVX before trusting AVX feature bits
    if (f.avx && !os_supports_avx()) {
        f.avx = false;
    }

    if (max_leaf >= 7) {
        // Leaf 7, sub-leaf 0: extended features
        __cpuid_count(7, 0, eax, ebx, ecx, edx);

        f.bmi1     = (ebx >> 3) & 1;
        f.avx2     = (ebx >> 5) & 1;
        f.bmi2     = (ebx >> 8) & 1;
        f.avx512f  = (ebx >> 16) & 1;
        f.avx512dq = (ebx >> 17) & 1;
        f.avx512ifma = (ebx >> 21) & 1;
        f.avx512bw = (ebx >> 30) & 1;
        f.avx512vl = (ebx >> 31) & 1;

        f.avx512vbmi  = (ecx >> 1) & 1;
        f.avx512vbmi2 = (ecx >> 6) & 1;
        f.avx512vnni  = (ecx >> 11) & 1;
        f.avx512bitalg   = (ecx >> 12) & 1;
        f.avx512vpopcntdq = (ecx >> 14) & 1;

        f.avxvnni  = (eax >> 4) & 1;  // AVX-VNNI is in leaf 7, sub-leaf 1
        // Actually AVX-VNNI is in sub-leaf 1
        __cpuid_count(7, 1, eax, ebx, ecx, edx);
        f.avxvnni = (eax >> 4) & 1;
    }

    // Clear AVX2+ features if OS doesn't support AVX
    if (!f.avx) {
        f.avx2 = false;
        f.bmi2 = false;
        f.avxvnni = false;
        f.avx512f = false;
        f.avx512bw = false;
        f.avx512dq = false;
        f.avx512vl = false;
        f.avx512vnni = false;
        f.avx512vbmi = false;
        f.avx512vbmi2 = false;
        f.avx512bitalg = false;
        f.avx512vpopcntdq = false;
        f.avx512ifma = false;
    }

    // Clear AVX-512 features if OS doesn't support AVX-512
    if (f.avx512f && !os_supports_avx512()) {
        f.avx512f = false;
        f.avx512bw = false;
        f.avx512dq = false;
        f.avx512vl = false;
        f.avx512vnni = false;
        f.avx512vbmi = false;
        f.avx512vbmi2 = false;
        f.avx512bitalg = false;
        f.avx512vpopcntdq = false;
        f.avx512ifma = false;
    }

    // Detect AMD Zen 1/2 (slow pdep/pext)
    if (f.is_amd && f.bmi2) {
        uint32_t family = ((eax >> 8) & 0xf);
        uint32_t ext_family = ((eax >> 20) & 0xff);

        // Re-read leaf 1 for family info
        __cpuid(1, eax, ebx, ecx, edx);
        family = ((eax >> 8) & 0xf);
        ext_family = ((eax >> 20) & 0xff);
        uint32_t full_family = family + ext_family;

        // Zen 1/Zen+ is family 0x17 (23), Zen 2 is also family 0x17
        // Zen 3 is family 0x19 (25)
        if (full_family == 0x17) {
            f.is_zen1_2 = true;
        }
    }

    return f;
}

using EntryFunc = int (*)(int, char*[]);

struct ArchEntry {
    const char* name;
    EntryFunc   func;
};

// FAT_ARCH_TABLE_BEGIN
// Order from best to worst -- first match wins.
// make_fat.py controls which entries are included via #ifdef.
static const ArchEntry arch_table[] = {
#ifdef HAS_X86_64_AVX512ICL
    {"x86-64-avx512icl", sf_main_x86_64_avx512icl},
#endif
#ifdef HAS_X86_64_VNNI512
    {"x86-64-vnni512", sf_main_x86_64_vnni512},
#endif
#ifdef HAS_X86_64_AVX512
    {"x86-64-avx512", sf_main_x86_64_avx512},
#endif
#ifdef HAS_X86_64_AVXVNNI
    {"x86-64-avxvnni", sf_main_x86_64_avxvnni},
#endif
#ifdef HAS_X86_64_BMI2
    {"x86-64-bmi2", sf_main_x86_64_bmi2},
#endif
#ifdef HAS_X86_64_AVX2
    {"x86-64-avx2", sf_main_x86_64_avx2},
#endif
#ifdef HAS_X86_64_SSE41_POPCNT
    {"x86-64-sse41-popcnt", sf_main_x86_64_sse41_popcnt},
#endif
#ifdef HAS_X86_64_SSSE3
    {"x86-64-ssse3", sf_main_x86_64_ssse3},
#endif
#ifdef HAS_X86_64_SSE3_POPCNT
    {"x86-64-sse3-popcnt", sf_main_x86_64_sse3_popcnt},
#endif
#ifdef HAS_X86_64
    {"x86-64", sf_main_x86_64},
#endif
    {nullptr, nullptr},
};
// FAT_ARCH_TABLE_END

static bool arch_supported(const char* name, const CpuFeatures& f) {
    if (strcmp(name, "x86-64-avx512icl") == 0)
        return f.avx512f && f.avx512bw && f.avx512dq && f.avx512vl && f.avx512vnni
            && f.avx512vbmi && f.avx512vbmi2 && f.avx512bitalg && f.avx512vpopcntdq
            && f.avx512ifma && f.bmi2;
    if (strcmp(name, "x86-64-vnni512") == 0)
        return f.avx512f && f.avx512bw && f.avx512dq && f.avx512vl && f.avx512vnni
            && f.bmi2;
    if (strcmp(name, "x86-64-avx512") == 0)
        return f.avx512f && f.avx512bw && f.avx512dq && f.avx512vl && f.bmi2;
    if (strcmp(name, "x86-64-avxvnni") == 0)
        return f.avx2 && f.avxvnni && f.bmi2;
    if (strcmp(name, "x86-64-bmi2") == 0)
        return f.avx2 && f.bmi2 && !f.is_zen1_2;
    if (strcmp(name, "x86-64-avx2") == 0)
        return f.avx2;
    if (strcmp(name, "x86-64-sse41-popcnt") == 0)
        return f.sse41 && f.popcnt;
    if (strcmp(name, "x86-64-ssse3") == 0)
        return f.ssse3;
    if (strcmp(name, "x86-64-sse3-popcnt") == 0)
        return f.sse3 && f.popcnt;
    if (strcmp(name, "x86-64") == 0)
        return f.sse2;  // x86-64 baseline always has SSE2
    return false;
}

int main(int argc, char* argv[]) {
    CpuFeatures features = detect_cpu();

    for (int i = 0; arch_table[i].name != nullptr; ++i) {
        if (arch_supported(arch_table[i].name, features)) {
            return arch_table[i].func(argc, argv);
        }
    }

    fprintf(stderr, "Error: No compatible Stockfish build found for this CPU.\n");
    return 1;
}
