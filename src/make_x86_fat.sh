
rm -r objs/baseline objs/sse41-popcnt objs/avx2 objs/bmi2 objs/avx512 objs/avx512vnni objs/avx512icl objs/avxvnni

mkdir objs/baseline
mkdir objs/sse41-popcnt
mkdir objs/avx2
mkdir objs/bmi2
mkdir objs/avx512
mkdir objs/avx512vnni
mkdir objs/avx512icl
mkdir objs/avxvnni

set -e
CXXFLAGS=-DStockfish=StockfishBaseline make -j profile-build ARCH=x86-64
mv *.o PGOBENCH.out objs/baseline
CXXFLAGS=-DStockfish=StockfishSSE41Popcnt make -j profile-build ARCH=x86-64-sse41-popcnt
mv *.o PGOBENCH.out objs/sse41-popcnt
CXXFLAGS=-DStockfish=StockfishAVX2 make -j profile-build ARCH=x86-64-avx2
mv *.o PGOBENCH.out objs/avx2
CXXFLAGS=-DStockfish=StockfishBMI2 make -j profile-build ARCH=x86-64-bmi2
mv *.o PGOBENCH.out objs/bmi2
CXXFLAGS=-DStockfish=StockfishAVX512 make -j profile-build ARCH=x86-64-avx512
mv *.o PGOBENCH.out objs/avx512
CXXFLAGS=-DStockfish=StockfishAVX512VNNI make -j profile-build ARCH=x86-64-vnni512
mv *.o PGOBENCH.out objs/avx512vnni
CXXFLAGS=-DStockfish=StockfishAVX512ICL make -j profile-build ARCH=x86-64-avx512icl
mv *.o PGOBENCH.out objs/avx512icl
CXXFLAGS=-DStockfish=StockfishAVXVNNI make -j profile-build ARCH=x86-64-avxvnni
mv *.o PGOBENCH.out objs/avxvnni