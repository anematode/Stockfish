from subprocess import Popen, PIPE, check_call
from pathlib import Path

import shutil
import os
import re

VERBOSE=True

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "src")
BUILDS_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "builds")

def top_level_only(l: list[str]):
	s = set()
	for name in l:
		s.add(n

# Get list of relevant source files + .nnue files
files = check_call(["git", "ls-files"], cwd=SRC_DIR).splitlines()
files = top_level_only(files)
print(files)
exit(0)

class Arch:
	def __init__(self, name: str):
		self.name = name
		# e.g. Stockfish_x86_64_avx2
		self.namespace = "Stockfish_" + name.replace("-","_")
		kself.build_dir = os.path.join(BUILDS_DIR, self.name)

	def set_up(self):
		# delete the build dir if it already exists, then create a src directory
		# and symlink all non-gitignored files, as well as NNUE files, to the main src
		try:
			shutil.rmtree(self.build_dir)
		except FileNotFoundError:
			pass
		os.makedirs(self.build_dir, exist_ok=True)

		directory = Path(SRC_DIR)
		for item in directory.iterdir():
			if item.is_dir() or include_file(item.name):
				os.symlink(os.path.join(self.build_dir, item.name), item)
			print(item, item.is_dir())
		
	name: str
	namespace: str
	build_dir: str


def read_arches(s: str):
	matches = re.findall(r"x86-64-[a-z0-9]{4,}", s)
	matches.append("x86-64")
	matches = [*set(matches)]
	if VERBOSE:
		joined = ",".join(matches)
		print(f"Found {len(matches)} arches: {joined}");
	return [*map(Arch, matches)]

def run_make(path: str, args: list[str]):
	if VERBOSE:
		print("Running: make " + " ".join(args))
	env = os.environ.copy()
	check_call(["make", *args], env=env, cwd=path)

file_path = os.path.realpath(__file__)
arches = read_arches(open(os.path.join(SRC_DIR, "Makefile"), "r").read())

# First run make net in the main src directory to ensure the nets are downloaded.

run_make(SRC_DIR, ["net"])

for arch in arches:
	arch.set_up()
