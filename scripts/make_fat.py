from subprocess import Popen, PIPE, check_call
from pathlib import Path

import shutil
import os
import re

VERBOSE=True

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "src")
BUILDS_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "builds")

class Arch:
	def __init__(self, name: str):
		self.name = name
		# e.g. Stockfish_x86_64_avx2
		self.namespace = "Stockfish_" + name.replace("-","_")
		self.build_dir = os.path.join(os.path.dirname(BUILDS_DIR, self.name))

	def set_up():
		# delete the build dir if it already exists, then create a src directory
		# and symlink all non-gitignored files, as well as NNUE files, to the main src
		shutil.rmtree(self.build_dir)
		os.makedirs(self.build_dir, exist_ok=True)

		directory = Path(SRC_DIR)
		for item in directory.iterdir():
			if item.is_dir():
				os.symlink
			else:
				item.unlink()
		
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
	return matches

def run_make(path: str, args: list[str]):
	if VERBOSE:
		print("Running: make " + " ".join(args))
	env = os.environ.copy()
	check_call(["make", *args], env=env)

file_path = os.path.realpath(__file__)
arches = read_arches(open(os.path.join(SRC_DIR, "Makefile"), "r").read())

# First run make clean; make -j build in the root src directory so as to ensure
# the requisite files are downloaded.

run_make(SRC_DIR, ["clean"])
run_make(SRC_DIR, ["-j", "build"])

for arch in arches:
	setup_build_dir(
