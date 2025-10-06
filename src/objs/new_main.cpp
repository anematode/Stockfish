#include <cpuid.h>
#include <iostream>

namespace StockfishBaseline { extern int main(int argc, char* argv[]); }
namespace StockfishSSE41Popcnt { extern int main(int argc, char* argv[]); }
namespace StockfishAVX2 { extern int main(int argc, char* argv[]); }
namespace StockfishBMI2 { extern int main(int argc, char* argv[]); }
namespace StockfishAVXVNNI { extern int main(int argc, char* argv[]); }
namespace StockfishAVX512 { extern int main(int argc, char* argv[]); }
namespace StockfishAVX512VNNI { extern int main(int argc, char* argv[]); }
namespace StockfishAVX512ICL { extern int main(int argc, char* argv[]); }

int main(int argc, char *argv[]) {
    unsigned max_leaf;
    __get_cpuid_max(0, &max_leaf);
    if (max_leaf < 1U)
        return 1;
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    if (max_leaf < 7U || !(ecx & (1 << 19)) || !(ecx & (1 << 23))) { // no popcnt or no sse4.1
        std::cout << "Selected baseline build\n";
        return StockfishBaseline::main(argc, argv);
    }

    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    if (!(ebx & (1 << 5))) { // no avx2
        std::cout << "Selected sse4.1/popcnt build\n";
        return StockfishSSE41Popcnt::main(argc, argv);
    }
    if (!(ebx & (1 << 8))) { // no bmi2 (todo, detect slow Zen 2)
        std::cout << "Selected avx2 build\n";
        return StockfishAVX2::main(argc, argv);
    }
    if (!(ebx & (1 << 16)) || !(ebx & (1 << 31)) || !(ebx & (1 << 30))) { // no avx512f/vl/bw
        __get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx);
        if (eax & (1 << 4)) {  // avxvnni
            std::cout << "Selected avxvnni build\n";
            return StockfishAVXVNNI::main(argc, argv);
        } else {
            std::cout << "Selected bmi2 build\n";
            return StockfishBMI2::main(argc, argv);
        }
    }
    if (!(ecx & 1 << 11 /* vnni512 */)) {
        std::cout << "Selected avx512 build\n";
        return StockfishAVX512::main(argc, argv);
    }
    if (!(ebx & 1 << 21 /* ifma */) || !(ecx & 1 << 1 /* vbmi */) || !(ecx & 1 << 6 /* vbmi2 */) || !(ecx & 1 << 14 /* vpopcntdq */)
        || !(ecx & 1 << 12 /* bitalg */) || !(ecx & 1 << 10 /* vpclmulqdq */) || !(ecx & 1 << 8 /* gfni */) || !(ecx & 1 << 9 /* vaes */)) {
        std::cout << "Selected vnni512 build\n";
        return StockfishAVX512VNNI::main(argc, argv);
    }
    std::cout << "Selected avx512icl build\n";
    return StockfishAVX512ICL::main(argc, argv);
}