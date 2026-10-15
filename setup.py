r"""
Build the bundled ``hypellfrob`` driver as part of installing the package.

``sage -pip install git+<url>`` should leave nothing to do by hand, so the
C++ in ``hypellfrob-threaded`` is compiled here and the resulting ``euler``
binary is put inside the package, where
:func:`hyperell_regulator.frobenius.euler_binary` looks for it first.

A private, thread-safe NTL is built from a pinned upstream download and linked
statically into euler. GMP comes from Sage's prefix (GMP_INC/GMP_LIB override
its include/library directories). Sage's own NTL is never modified or linked.

The build is *not* required: without a compiler, or without GMP, the install
still succeeds and :mod:`hyperell_regulator.frobenius` falls back to Sage's
own Frobenius matrices, which are correct and slower.  Only the speed is
lost, so a failure here warns rather than raises.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "hypellfrob-threaded"
NTL_VERSION = "11.6.0"
NTL_SHA256 = "30e29eb21c5da6db031fc7dc5b1a6c728f226c7128e922421247157bc89ecd86"
CHECK = ROOT / "hyperell_regulator" / "frobenius.py"


def _prefixes():
    """Where to look for GMP, best guess first."""
    out = []
    for var in ("SAGE_LOCAL", "CONDA_PREFIX"):
        if os.environ.get(var):
            out.append(os.environ[var])
    if os.environ.get("SAGE_ROOT"):
        out.append(os.path.join(os.environ["SAGE_ROOT"], "local"))
    for prefix in (sys.prefix, sys.base_prefix):
        out.append(prefix)
        # Sage's Python can live in local/var/lib/sage/venv-python*.
        parent = prefix
        while parent != os.path.dirname(parent):
            parent = os.path.dirname(parent)
            if os.path.basename(parent) == "local":
                out.append(parent)
    out.extend(("/opt/homebrew", "/usr/local", "/usr"))
    return list(dict.fromkeys(out))


def _gmp_paths():
    """Find Sage's GMP, preserving explicit include/library overrides."""
    inc, lib = os.environ.get("GMP_INC"), os.environ.get("GMP_LIB")
    if inc and lib:
        return inc, lib
    for prefix in map(Path, _prefixes()):
        if (prefix / "include" / "gmp.h").is_file():
            return inc or str(prefix / "include"), lib or str(prefix / "lib")
    raise RuntimeError("GMP headers not found; set GMP_INC and GMP_LIB")


def _build_jobs():
    return max(1, int(os.environ.get("HYPERELL_REGULATOR_BUILD_JOBS",
                                     os.cpu_count() or 1)))


class build_with_euler(build_py):
    def run(self):
        super().run()
        target = Path(self.build_lib) / "hyperell_regulator" / "bin" / "euler"
        # A failed rebuild must not package an accelerator from an older build.
        target.unlink(missing_ok=True)
        try:
            binary = self._make()
        except Exception as exc:                       # noqa: BLE001
            if os.environ.get("HYPERELL_REGULATOR_REQUIRE_EULER") == "1":
                raise
            print("hyperell-regulator: could not build the hypellfrob driver "
                  "(%s); falling back to Sage's Frobenius matrices" % exc)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(binary, target)
        target.chmod(0o755)
        print("hyperell-regulator: built %s" % target)

    def _build_ntl(self, directory, gmp_inc, gmp_lib, jobs):
        archive = Path(directory) / ("ntl-%s.tar.gz" % NTL_VERSION)
        url = "https://github.com/libntl/ntl/archive/refs/tags/v%s.tar.gz" % NTL_VERSION
        print("hyperell-regulator: downloading " + url, flush=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "hyperell-regulator/0.1 (NTL build)"})
        with urllib.request.urlopen(request, timeout=60) as response, \
                archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != NTL_SHA256:
            raise RuntimeError("Downloaded NTL source checksum mismatch")
        # Only extract the exact, pinned upstream archive verified above.
        with tarfile.open(archive) as source:
            source.extractall(directory)
        root = Path(directory) / ("ntl-" + NTL_VERSION)
        subprocess.run([
            "./configure", "CXX=" + os.environ.get("CXX", "c++"),
            "CXXFLAGS=" + os.environ.get("CXXFLAGS", "-O2 -std=c++17"),
            "NTL_THREADS=on", "NTL_THREAD_BOOST=on", "NTL_GMP_LIP=on",
            "NTL_GF2X_LIB=off", "SHARED=off", "NATIVE=off", "TUNE=generic",
            "GMP_INCDIR=" + gmp_inc, "GMP_LIBDIR=" + gmp_lib,
        ], cwd=root / "src", check=True)
        subprocess.run(["make", "-j" + str(jobs)], cwd=root / "src", check=True)
        return root

    def _make(self):
        inc, lib = _gmp_paths()
        jobs = _build_jobs()
        build = Path(self.build_lib).resolve().parent / "euler-build"
        build.mkdir(parents=True, exist_ok=True)
        # Fresh objects and headers prevent ABI mixing with previous NTL builds.
        with tempfile.TemporaryDirectory(prefix="threaded-", dir=build) as directory:
            ntl = self._build_ntl(directory, inc, lib, jobs)
            objects = Path(directory) / "driver"
            args = ["make", "-j" + str(jobs), "-C", str(SRC),
                    "CONFIG_MK=", "BUILD=" + str(objects),
                    "CXX=" + os.environ.get("CXX", "c++"),
                    "NTL_INC=" + str(ntl / "include"),
                    "NTL_LIB=" + str(ntl / "src"),
                    "NTL_LINK=" + str(ntl / "src" / "ntl.a"), "NTL_RPATH=",
                    "GMP_INC=" + inc, "GMP_LIB=" + lib]
            subprocess.run(args, check=True)
            binary = str(objects / "euler")
            # Reuse the installed check without importing Sage in build isolation.
            subprocess.run([sys.executable, str(CHECK), "--check", "--binary", binary],
                           check=True, timeout=120)
            shutil.copy(binary, build / "euler")
        return str(build / "euler")


class binary_distribution(Distribution):
    """The wheel holds a compiled program, so it is not ``py3-none-any``."""

    def has_ext_modules(self):
        return True


setup(cmdclass={"build_py": build_with_euler}, distclass=binary_distribution)
