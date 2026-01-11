from subprocess import Popen, PIPE, check_output, check_call
import sys
from pathlib import Path

import shutil
import os
import re

VERBOSE=True

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "src")
BUILDS_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "builds")

final_args = "-flto -flto-partition=one -flto=jobserver -funroll-loops -fno-exceptions -O3 -funroll-loops -fno-peel-loops -fno-tracer -lpthread -lrt"
final_args = final_args.split(' ')

def safe_rmtree(path: str):
	# Verify that the path is a subdirectory of BUILDS_DIR
	if not path.startswith(BUILDS_DIR):
		raise ValueError("Tried to remove path " + path)
	try:
		shutil.rmtree(path)
	except FileNotFoundError:
		pass

def top_level_only(l: list[str]):
	s = set()
	for name in l:
		if '/' in name:
			s.add(name[:name.index('/')])
		else:
			s.add(name)
	return [*s]

class Arch:
	def __init__(self, name: str):
		self.name = name
		self.suffix = name.replace("-","_");
		# e.g. Stockfish_x86_64_avx2
		self.namespace = "Stockfish_" + self.suffix
		self.build_dir = os.path.join(BUILDS_DIR, self.name)

	def set_up(self, filelist: list[str]):
		# delete the build dir if it already exists, then create a src directory
		# and symlink all non-gitignored files, as well as NNUE files, to the main src
		safe_rmtree(self.build_dir)
		os.makedirs(self.build_dir, exist_ok=False)
		script_link = os.path.join(BUILDS_DIR, "scripts")
		if not os.path.exists(script_link):
			os.symlink(SCRIPTS_DIR, script_link)

		for item in filelist:
			original = os.path.join(SRC_DIR, item)
			os.symlink(original, os.path.join(self.build_dir, item))

	def make(self, make_threads: int):
		env = {}
		env["ARCH"] = self.name
		env["CXXFLAGS"] = f"-ffat-lto-objects -DStockfish={self.namespace}"
		return run_make(self.build_dir, [f"-j{make_threads}", "fat-object"], additional_env=env)

	def get_merged_object(self):
		return os.path.join(self.build_dir, "stockfish.o")

	def adjust_stockfish_o(self):
		path = self.get_merged_object()
		assert os.path.exists(path)

		weaken_symbols = ["gEmbeddedNNUEBigEnd", "gEmbeddedNNUEBigSize", "gEmbeddedNNUESmallEnd", "gEmbeddedNNUESmallSize", "gEmbeddedNNUESmallStart", "gEmbeddedNNUEBigStart"]
		weaken = []
		for i in weaken_symbols:
			weaken.append("--weaken-symbol")
			weaken.append(i)

		cmd = ["objcopy", "--rename-section", f".init_array=.{self.suffix}_init", "--localize-symbol", "main", "--localize-symbol", ".gnu.lto_main*", *weaken, path]
		check_output(cmd, cwd=self.build_dir)
		
	name: str
	suffix: str
	namespace: str
	build_dir: str


def read_arches(s: str) -> list[Arch]:
	matches = re.findall(r"x86-64-[a-z0-9-]{4,}", s)
	matches.append("x86-64")
	matches = [*set(matches)]
	if VERBOSE:
		joined = ",".join(matches)
		print(f"Found {len(matches)} arches: {joined}");
	matches.remove("x86-64-modern")
	return [*map(Arch, matches)]

def run_make(path: str, args: list[str], additional_env=None):
	if additional_env is None:
		additional_env = {}
	if VERBOSE:
		print("Running: make " + " ".join(args))
	env = os.environ.copy() | additional_env
	p = Popen(["make", *args], stdout=PIPE, env=env, cwd=path)
	return p

file_path = os.path.realpath(__file__)
arches = read_arches(open(os.path.join(SRC_DIR, "Makefile"), "r").read())

if len(sys.argv) > 1 and sys.argv[1] == "print":
	for arch in arches:
		print("DEFINE_BUILD(" + arch.suffix + ")")

# First run make net in the main src directory to ensure the nets are downloaded.
# From its output, get the net files.
nnue_list = run_make(SRC_DIR, ["net"]).communicate()[0]
nnue_list = re.findall(R"\S+.nnue", nnue_list.decode('utf-8'))

# Get list of relevant source files + .nnue files
files = check_output(["git", "ls-files"], cwd=SRC_DIR).decode('utf-8').splitlines()
files = top_level_only(files)
files += nnue_list

if VERBOSE:
	print("Linking files: ", files)

for arch in arches:
	arch.set_up(files)

concurrency = os.process_cpu_count()
make_threads = max(concurrency // len(arches), 1)

jobs: list[Popen] = []
for arch in arches:
	jobs.append(arch.make(make_threads))
for job in jobs:
	out, err = job.communicate()
	assert not job.returncode

for arch in arches:
	arch.adjust_stockfish_o()
