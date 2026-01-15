#include <cpuid.h>
#include <stdio.h>

#define DEFINE_BUILD(x) \
namespace Stockfish_##x { extern int main(int argc, char* argv[]); } \
extern "C" void (*__start_##x##_init[])(void); \
extern "C" void (*__stop_##x##_init[])(void); \
 int entry_##x(int argc, char* argv[]) {\
 puts("Selected: " #x); \
 unsigned count = __stop_##x##_init - __start_##x##_init; \
    for (unsigned i = 0; i < count; i++) { \
        __start_##x##_init[i](); \
    } \
  return Stockfish_##x::main(argc, argv); \
}

DEFINE_BUILD(x86_64_bmi2)
DEFINE_BUILD(x86_64_sse3_popcnt)
DEFINE_BUILD(x86_64_sse41_popcnt)
DEFINE_BUILD(x86_64_vnni512)
DEFINE_BUILD(x86_64_ssse3)
DEFINE_BUILD(x86_64_avx512)
DEFINE_BUILD(x86_64)
DEFINE_BUILD(x86_64_avxvnni)
DEFINE_BUILD(x86_64_avx2)
DEFINE_BUILD(x86_64_avx512icl)

int main(int argc, char *argv[]) {
    unsigned max_leaf;
    __get_cpuid_max(0, &max_leaf);
    if (max_leaf < 1U)
        return 1;
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    if (max_leaf < 7U || !(ecx & (1 << 19)) || !(ecx & (1 << 23))) { // no popcnt or no sse4.1
        return entry_x86_64(argc, argv);
    }

    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    if (!(ebx & (1 << 5))) { // no avx2
        return entry_x86_64_sse41_popcnt(argc, argv);
    }

    if (!(ebx & (1 << 8))) { // no bmi2 (todo, detect slow Zen 2)
        return entry_x86_64_avx2(argc, argv);
    }

    if (!(ebx & (1 << 16)) || !(ebx & (1 << 31)) || !(ebx & (1 << 30))) { // no avx512f/vl/bw
        __get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx);
        if (eax & (1 << 4)) {  // avxvnni
            return entry_x86_64_avxvnni(argc, argv);
        } else {
            return entry_x86_64_bmi2(argc, argv);
        }
    }

    if (!(ecx & 1 << 11 /* vnni512 */)) {
        return entry_x86_64_avx512(argc, argv);
    }

    if (!(ebx & 1 << 21 /* ifma */) || !(ecx & 1 << 1 /* vbmi */) || !(ecx & 1 << 6 /* vbmi2 */) || !(ecx & 1 << 14 /* vpopcntdq */)
        || !(ecx & 1 << 12 /* bitalg */) || !(ecx & 1 << 10 /* vpclmulqdq */) || !(ecx & 1 << 8 /* gfni */) || !(ecx & 1 << 9 /* vaes */)) {
        return entry_x86_64_vnni512(argc, argv);
    }

    return entry_x86_64_avx512icl(argc, argv);
}
