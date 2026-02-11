#!/usr/bin/env python3
"""
Build a Stockfish "fat binary" for x86-64 Linux.

The fat binary contains multiple architecture-specific builds of Stockfish,
with a small dispatcher that uses cpuid to detect CPU features at startup
and jumps to the best available build.

Usage:
    python3 make_fat.py [options]

The script:
  1. Builds each architecture with PGO + LTO (using make profile-build).
  2. Extracts the LTO-resolved native code from each build.
  3. Renames 'main' to 'sf_main_<arch>' and localizes all other symbols.
  4. Builds the NNUE embedding once (shared across all architectures).
  5. Compiles the fat_main.cpp dispatcher.
  6. Links everything into a single fat binary.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

# x86-64 architectures, ordered from most to least capable.
# Each entry: (arch_name, make_arch_name, symbol_suffix)
ALL_X86_64_ARCHS = [
    ("x86-64-avx512icl", "x86-64-avx512icl", "x86_64_avx512icl"),
    ("x86-64-vnni512",   "x86-64-vnni512",   "x86_64_vnni512"),
    ("x86-64-avx512",    "x86-64-avx512",    "x86_64_avx512"),
    ("x86-64-avxvnni",   "x86-64-avxvnni",   "x86_64_avxvnni"),
    ("x86-64-bmi2",      "x86-64-bmi2",      "x86_64_bmi2"),
    ("x86-64-avx2",      "x86-64-avx2",      "x86_64_avx2"),
    ("x86-64-sse41-popcnt", "x86-64-sse41-popcnt", "x86_64_sse41_popcnt"),
    ("x86-64-ssse3",     "x86-64-ssse3",     "x86_64_ssse3"),
    ("x86-64-sse3-popcnt", "x86-64-sse3-popcnt", "x86_64_sse3_popcnt"),
    ("x86-64",           "x86-64",           "x86_64"),
]

# Default set for building (excludes archs needing avx512/avx-vnni hardware)
DEFAULT_ARCHS = [
    "x86-64-bmi2",
    "x86-64-avx2",
    "x86-64-sse41-popcnt",
    "x86-64-ssse3",
    "x86-64-sse3-popcnt",
    "x86-64",
]

# NNUE embedding symbols that must remain global across all arch builds
NNUE_GLOBAL_SYMBOLS = [
    "gEmbeddedNNUEBigData",
    "gEmbeddedNNUEBigEnd",
    "gEmbeddedNNUEBigSize",
    "gEmbeddedNNUESmallData",
    "gEmbeddedNNUESmallEnd",
    "gEmbeddedNNUESmallSize",
]


def create_nnue_stub(build_dir, comp):
    """
    Create a stub object file with dummy NNUE symbol definitions.
    This is used during per-arch intermediate linking so that the linker
    doesn't fail on undefined NNUE symbols. The real definitions are
    provided at the final fat binary link step.
    """
    stub_src = os.path.join(build_dir, "nnue_stub.c")
    stub_obj = os.path.join(build_dir, "nnue_stub.o")

    with open(stub_src, "w") as f:
        f.write("""\
/* Stub NNUE symbol definitions for fat binary intermediate linking. */
const unsigned char        gEmbeddedNNUEBigData[1]   = {0};
const unsigned char* const gEmbeddedNNUEBigEnd       = &gEmbeddedNNUEBigData[1];
const unsigned int         gEmbeddedNNUEBigSize      = 1;
const unsigned char        gEmbeddedNNUESmallData[1] = {0};
const unsigned char* const gEmbeddedNNUESmallEnd     = &gEmbeddedNNUESmallData[1];
const unsigned int         gEmbeddedNNUESmallSize    = 1;
""")

    cc = "gcc" if comp in ("gcc", "mingw") else comp
    run([cc, "-O2", "-c", "-m64", stub_src, "-o", stub_obj])
    return stub_obj


def run(cmd, cwd=None, env=None, check=True):
    """Run a command, printing it for debugging."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}", file=sys.stderr)
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def find_arch_info(arch_name):
    """Look up architecture info by name."""
    for name, make_arch, suffix in ALL_X86_64_ARCHS:
        if name == arch_name:
            return name, make_arch, suffix
    raise ValueError(f"Unknown architecture: {arch_name}")


def setup_arch_build_dir(src_dir, build_dir, arch_name):
    """
    Set up a build directory for one architecture.
    Creates a structure like:
        build_dir/arch_<name>/src/   <- source copy (Makefile here)
        build_dir/arch_<name>/scripts/ <- scripts (for net.sh)
    This mirrors the repo layout so that relative paths in the Makefile work.
    """
    arch_top = os.path.join(build_dir, f"arch_{arch_name}")
    arch_src_dir = os.path.join(arch_top, "src")
    arch_scripts_dir = os.path.join(arch_top, "scripts")
    os.makedirs(arch_src_dir, exist_ok=True)

    # Copy source files
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(arch_src_dir, item)
        if os.path.isdir(src_path):
            if item in ('.git', '__pycache__'):
                continue
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        elif os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    # Copy scripts directory (needed for net.sh, referenced as ../scripts/)
    repo_root = os.path.dirname(src_dir)
    scripts_src = os.path.join(repo_root, "scripts")
    if os.path.exists(scripts_src):
        if os.path.exists(arch_scripts_dir):
            shutil.rmtree(arch_scripts_dir)
        shutil.copytree(scripts_src, arch_scripts_dir)

    return arch_src_dir


def build_arch(src_dir, build_dir, arch_name, make_arch, jobs, comp):
    """
    Build one architecture with PGO + LTO, producing LTO-resolved native .o files.

    Uses -save-temps to capture the .ltrans*.ltrans.o files that contain
    the final LTO-resolved native code.
    """
    arch_build_dir = setup_arch_build_dir(src_dir, build_dir, arch_name)

    # Create NNUE stub for intermediate linking
    create_nnue_stub(arch_build_dir, comp)
    stub_basename = "nnue_stub.o"

    # Clean first
    run(["make", "clean"], cwd=arch_build_dir)
    # Re-create stub since clean removed .o files
    create_nnue_stub(arch_build_dir, comp)

    # Pass our flags via environment variables so they persist through
    # the PGO make targets (which override EXTRACXXFLAGS/EXTRALDFLAGS).
    # ENV_CXXFLAGS and ENV_LDFLAGS are captured at MAKELEVEL=0 and
    # prepended to all subsequent compile/link commands.
    env = os.environ.copy()
    env["CXXFLAGS"] = "-DNNUE_EMBEDDING_EXTERN -no-pie"
    env["LDFLAGS"] = f"-no-pie {os.path.join(arch_build_dir, stub_basename)}"

    # Build with PGO + LTO
    print(f"\n{'='*60}")
    print(f"Building {arch_name} with PGO + LTO...")
    print(f"{'='*60}")

    # Step 1: Build instrumented executable
    print(f"\n  Step 1/4: Building instrumented executable for {arch_name}...")
    run(["make", f"-j{jobs}", f"ARCH={make_arch}", f"COMP={comp}",
         "net", "config-sanity", "objclean", "profileclean"],
        cwd=arch_build_dir, env=env)
    # Re-create stub after profileclean
    create_nnue_stub(arch_build_dir, comp)

    # Determine profile method based on compiler
    gcc_is_clang = False
    if comp == "gcc":
        result = run(["g++", "--version"], check=False)
        if "clang" in result.stdout:
            gcc_is_clang = True

    if comp == "clang" or gcc_is_clang:
        profile_make = "clang-profile-make"
        profile_use = "clang-profile-use"
    elif comp == "icx":
        profile_make = "icx-profile-make"
        profile_use = "icx-profile-use"
    else:
        profile_make = "gcc-profile-make"
        profile_use = "gcc-profile-use"

    run(["make", f"-j{jobs}", f"ARCH={make_arch}", f"COMP={comp}",
         profile_make],
        cwd=arch_build_dir, env=env)

    # Step 2: Run benchmark for PGO
    print(f"\n  Step 2/4: Running benchmark for {arch_name}...")
    bench_result = run(["./stockfish", "bench"], cwd=arch_build_dir, check=False)
    # Write bench output for profile data
    with open(os.path.join(arch_build_dir, "PGOBENCH.out"), "w") as f:
        f.write(bench_result.stdout)
        f.write(bench_result.stderr)

    # Step 3: Build optimized executable with profile data
    print(f"\n  Step 3/4: Building optimized executable for {arch_name}...")
    run(["make", f"-j{jobs}", f"ARCH={make_arch}", f"COMP={comp}", "objclean"],
        cwd=arch_build_dir, env=env)
    # Re-create stub since objclean may have removed it
    create_nnue_stub(arch_build_dir, comp)

    # For the final build, also add -save-temps to get ltrans files
    env_final = env.copy()
    env_final["CXXFLAGS"] = "-DNNUE_EMBEDDING_EXTERN -save-temps -no-pie"
    env_final["LDFLAGS"] = f"-save-temps -no-pie {os.path.join(arch_build_dir, stub_basename)}"

    run(["make", f"-j{jobs}", f"ARCH={make_arch}", f"COMP={comp}",
         profile_use],
        cwd=arch_build_dir, env=env_final)

    # Find and preserve the LTO-resolved .ltrans.o files BEFORE cleaning.
    # profileclean deletes stockfish.*lt* which matches our ltrans files.
    ltrans_files_raw = sorted(glob.glob(os.path.join(arch_build_dir, "*.ltrans*.ltrans.o")))
    if not ltrans_files_raw:
        raise RuntimeError(
            f"No .ltrans.o files found for {arch_name}. "
            f"LTO may not have been applied. Check build output."
        )

    # Copy ltrans files to safe location
    ltrans_files = []
    for i, f in enumerate(ltrans_files_raw):
        safe_path = os.path.join(arch_build_dir, f"ltrans_saved_{i}.o")
        shutil.copy2(f, safe_path)
        ltrans_files.append(safe_path)

    print(f"  Found {len(ltrans_files)} LTO-resolved object file(s) for {arch_name}")

    # Step 4: Clean profile data
    print(f"\n  Step 4/4: Cleaning profile data for {arch_name}...")
    run(["make", f"ARCH={make_arch}", f"COMP={comp}", "profileclean"],
        cwd=arch_build_dir)

    return arch_build_dir, ltrans_files


def process_arch_objects(ltrans_files, arch_suffix, output_path):
    """
    Process LTO-resolved object files for one architecture:
    1. Merge multiple .ltrans.o files via ld -r (if needed)
    2. Rename main -> sf_main_<arch>
    3. Localize all symbols except the entry point and NNUE references
    """
    if len(ltrans_files) == 1:
        merged_obj = ltrans_files[0]
    else:
        # Merge multiple ltrans objects into one
        merged_obj = output_path + ".merged.o"
        run(["ld", "-r"] + ltrans_files + ["-o", merged_obj])

    # Build the objcopy command:
    # 1. Rename main -> sf_main_<arch>
    # 2. Keep only the entry point and NNUE symbols as global
    entry_sym = f"sf_main_{arch_suffix}"

    cmd = ["objcopy",
           f"--redefine-sym=main={entry_sym}",
           f"--keep-global-symbol={entry_sym}"]

    # Keep NNUE symbols global (they're undefined references to shared data)
    for sym in NNUE_GLOBAL_SYMBOLS:
        cmd.append(f"--keep-global-symbol={sym}")

    cmd.extend([merged_obj, output_path])
    run(cmd)

    # Verify the entry point exists
    result = run(["nm", output_path])
    if entry_sym not in result.stdout:
        raise RuntimeError(f"Entry point {entry_sym} not found in {output_path}")

    print(f"  Created arch object: {output_path} (entry: {entry_sym})")


def build_nnue_embed(src_dir, build_dir, jobs, comp):
    """
    Build the NNUE embedding object file.
    This is compiled once at the baseline x86-64 architecture.
    """
    embed_dir = os.path.join(build_dir, "nnue_embed")
    os.makedirs(embed_dir, exist_ok=True)

    # Copy necessary files
    for d in ["incbin"]:
        src = os.path.join(src_dir, d)
        dst = os.path.join(embed_dir, d)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # Copy NNUE files
    for nnue_file in glob.glob(os.path.join(src_dir, "nn-*.nnue")):
        shutil.copy2(nnue_file, embed_dir)

    # Read the NNUE file names from evaluate.h
    eval_h = os.path.join(src_dir, "evaluate.h")
    big_name = None
    small_name = None
    with open(eval_h) as fh:
        for line in fh:
            if "EvalFileDefaultNameBig" in line and "#define" in line:
                big_name = line.split('"')[1]
            if "EvalFileDefaultNameSmall" in line and "#define" in line:
                small_name = line.split('"')[1]
    if not big_name or not small_name:
        raise RuntimeError("Could not find NNUE file names in evaluate.h")

    # Create a minimal source file that only embeds the NNUE data
    embed_src = os.path.join(embed_dir, "nnue_embed.cpp")
    with open(embed_src, "w") as f:
        f.write(f"""\
#define INCBIN_SILENCE_BITCODE_WARNING
#include "incbin/incbin.h"

// Embed the NNUE network data -- shared across all architecture builds.
INCBIN(EmbeddedNNUEBig, "{big_name}");
INCBIN(EmbeddedNNUESmall, "{small_name}");
""")

    # Compile at baseline x86-64 (most compatible)
    embed_obj = os.path.join(embed_dir, "nnue_embed.o")
    cxx = "g++" if comp == "gcc" else ("clang++" if comp == "clang" else comp)
    run([cxx, "-O2", "-c", "-m64", "-std=c++17",
         "-DNDEBUG",
         embed_src, "-o", embed_obj],
        cwd=embed_dir)

    print(f"  Built NNUE embedding: {embed_obj}")
    return embed_obj


def build_fat_main(src_dir, build_dir, archs, comp):
    """
    Compile the fat binary dispatcher (fat_main.cpp).
    """
    fat_main_src = os.path.join(src_dir, "fat_main.cpp")
    fat_main_obj = os.path.join(build_dir, "fat_main.o")

    # Build the -D flags for which architectures are included
    defines = []
    for arch_name, _, _ in archs:
        macro = "HAS_" + arch_name.upper().replace("-", "_")
        defines.extend([f"-D{macro}"])

    cxx = "g++" if comp == "gcc" else ("clang++" if comp == "clang" else comp)
    run([cxx, "-O2", "-c", "-m64", "-std=c++17",
         "-DNDEBUG", "-no-pie"] + defines +
        [fat_main_src, "-o", fat_main_obj])

    print(f"  Built fat_main dispatcher: {fat_main_obj}")
    return fat_main_obj


def link_fat_binary(arch_objects, nnue_obj, fat_main_obj, output_path, comp):
    """
    Link all components into the final fat binary.
    """
    cxx = "g++" if comp == "gcc" else ("clang++" if comp == "clang" else comp)

    cmd = [cxx, "-o", output_path, "-no-pie", "-m64",
           fat_main_obj, nnue_obj] + arch_objects + [
           "-lpthread", "-lrt",
           "-Wl,--no-as-needed"]

    run(cmd)

    # Strip the binary
    run(["strip", output_path])

    size = os.path.getsize(output_path)
    print(f"\n  Fat binary created: {output_path} ({size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build a Stockfish fat binary")
    parser.add_argument("--archs", nargs="+", default=None,
                        help=f"Architectures to include (default: {' '.join(DEFAULT_ARCHS)})")
    parser.add_argument("-j", "--jobs", type=int,
                        default=os.cpu_count() or 4,
                        help="Number of parallel build jobs")
    parser.add_argument("--comp", default="gcc",
                        choices=["gcc", "clang", "icx"],
                        help="Compiler to use (default: gcc)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output binary path (default: src/stockfish-fat)")
    parser.add_argument("--build-dir", default=None,
                        help="Build directory for intermediate files (default: tmpdir)")
    parser.add_argument("--no-pgo", action="store_true",
                        help="Skip PGO (faster build, less optimal)")
    args = parser.parse_args()

    # Determine source directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Handle both script in repo root and in src/
    if os.path.exists(os.path.join(script_dir, "src", "Makefile")):
        src_dir = os.path.join(script_dir, "src")
    elif os.path.exists(os.path.join(script_dir, "Makefile")):
        src_dir = script_dir
    else:
        print("Error: Cannot find src/Makefile. Run from the repository root.", file=sys.stderr)
        sys.exit(1)

    # Resolve architectures
    arch_names = args.archs if args.archs else DEFAULT_ARCHS
    archs = [find_arch_info(a) for a in arch_names]

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        output_path = os.path.join(src_dir, "stockfish-fat")

    # Create build directory
    if args.build_dir:
        build_dir = os.path.abspath(args.build_dir)
        os.makedirs(build_dir, exist_ok=True)
        cleanup_build_dir = False
    else:
        build_dir = tempfile.mkdtemp(prefix="sf_fat_")
        cleanup_build_dir = False  # Keep for debugging

    print(f"Source directory: {src_dir}")
    print(f"Build directory:  {build_dir}")
    print(f"Output:           {output_path}")
    print(f"Architectures:    {', '.join(a[0] for a in archs)}")
    print(f"Compiler:         {args.comp}")
    print(f"Jobs:             {args.jobs}")
    print(f"PGO:              {'no' if args.no_pgo else 'yes'}")
    print()

    # Step 1: Build NNUE embedding
    print("=" * 60)
    print("Building NNUE embedding (shared across all architectures)...")
    print("=" * 60)
    nnue_obj = build_nnue_embed(src_dir, build_dir, args.jobs, args.comp)

    # Step 2: Build each architecture
    arch_objects = []
    for arch_name, make_arch, arch_suffix in archs:
        if args.no_pgo:
            arch_build_dir, ltrans_files = build_arch_no_pgo(
                src_dir, build_dir, arch_name, make_arch, args.jobs, args.comp)
        else:
            arch_build_dir, ltrans_files = build_arch(
                src_dir, build_dir, arch_name, make_arch, args.jobs, args.comp)

        # Process the LTO-resolved objects
        arch_obj_path = os.path.join(build_dir, f"arch_{arch_suffix}.o")
        process_arch_objects(ltrans_files, arch_suffix, arch_obj_path)
        arch_objects.append(arch_obj_path)

    # Step 3: Build the fat_main dispatcher
    print("\n" + "=" * 60)
    print("Building fat binary dispatcher...")
    print("=" * 60)
    fat_main_obj = build_fat_main(src_dir, build_dir, archs, args.comp)

    # Step 4: Link everything together
    print("\n" + "=" * 60)
    print("Linking fat binary...")
    print("=" * 60)
    link_fat_binary(arch_objects, nnue_obj, fat_main_obj, output_path, args.comp)

    print("\nDone!")


def build_arch_no_pgo(src_dir, build_dir, arch_name, make_arch, jobs, comp):
    """
    Build one architecture WITHOUT PGO (but still with LTO).
    Used for faster testing.

    Note: LTO is enabled by default in the Makefile when optimize=yes
    and debug=no (the defaults). The -save-temps flag captures the
    LTO-resolved native code in .ltrans.o files.
    """
    arch_build_dir = setup_arch_build_dir(src_dir, build_dir, arch_name)

    # Create NNUE stub in the arch build directory
    nnue_stub = create_nnue_stub(arch_build_dir, comp)
    stub_basename = os.path.basename(nnue_stub)

    # Clean first
    run(["make", "clean"], cwd=arch_build_dir)
    # Re-create stub since clean may have removed it
    create_nnue_stub(arch_build_dir, comp)

    extra_cxx = "-DNNUE_EMBEDDING_EXTERN -save-temps -no-pie"
    extra_ld = f"-save-temps -no-pie {stub_basename}"

    print(f"\n{'='*60}")
    print(f"Building {arch_name} with LTO (no PGO)...")
    print(f"{'='*60}")

    run(["make", f"-j{jobs}", f"ARCH={make_arch}", f"COMP={comp}",
         f"EXTRACXXFLAGS={extra_cxx}",
         f"EXTRALDFLAGS={extra_ld}",
         "build"],
        cwd=arch_build_dir)

    # Find the LTO-resolved .ltrans.o files
    ltrans_files = sorted(glob.glob(os.path.join(arch_build_dir, "*.ltrans*.ltrans.o")))
    if not ltrans_files:
        raise RuntimeError(
            f"No .ltrans.o files found for {arch_name}. "
            f"LTO may not have been applied. Check build output."
        )

    print(f"  Found {len(ltrans_files)} LTO-resolved object file(s) for {arch_name}")
    return arch_build_dir, ltrans_files


if __name__ == "__main__":
    main()
