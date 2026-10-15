"""Private NTL builds and accelerator worker defaults."""
import io
import os
import runpy
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from setuptools import Distribution


def build_module():
    with patch('setuptools.setup'):
        return runpy.run_path(str(Path(__file__).resolve().parents[1] / 'setup.py'))


def test_sage_macos_venv_prefix():
    module = build_module()
    with patch.dict(os.environ, {}, clear=True), patch.object(
            module['sys'], 'prefix',
            '/private/var/tmp/sage/local/var/lib/sage/venv-python3.13'):
        assert '/private/var/tmp/sage/local' in module['_prefixes']()
        assert '/opt/homebrew' in module['_prefixes']()


def smoke_output():
    return "".join("%d 1 4 %s\n" % (p, " ".join(["0"] * 16))
                   for p in (101, 103, 107, 109))


def test_make_paths_and_smoke_test():
    module = build_module()
    cmd = module['build_with_euler'](Distribution())
    with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {'GMP_INC': '/sage/include', 'GMP_LIB': '/sage/lib',
                         'NTL_LINK': '-lunsafe'}, clear=True), \
            patch.object(cmd, '_build_ntl', return_value=Path('/private/ntl')), \
            patch('subprocess.run') as run, patch('shutil.copy'):
        cmd.build_lib = str(Path(directory) / 'lib')
        run.return_value.stdout = smoke_output()
        cmd._make()
    args = run.call_args_list[0].args[0]
    assert 'CONFIG_MK=' in args
    assert 'NTL_LINK=/private/ntl/src/ntl.a' in args
    assert 'NTL_INC=/private/ntl/include' in args
    assert 'GMP_LIB=/sage/lib' in args
    check = run.call_args_list[1]
    assert check.args[0][0] == module['sys'].executable
    assert check.args[0][2:4] == ['--check', '--binary']
    assert check.args[0][4].endswith('/driver/euler')
    assert check.kwargs['check'] is True
    assert check.kwargs['timeout'] == 120


def test_ntl_download_rejects_bad_checksum():
    module = build_module()
    cmd = module['build_with_euler'](Distribution())
    with tempfile.TemporaryDirectory() as directory, \
            patch('urllib.request.urlopen', return_value=io.BytesIO(b'corrupt')), \
            patch('tarfile.open') as extract, patch('subprocess.run') as run:
        with pytest.raises(RuntimeError, match='checksum mismatch'):
            cmd._build_ntl(directory, '/sage/include', '/sage/lib', 4)
        extract.assert_not_called()
        run.assert_not_called()


def test_accelerator_check_rejects_parallel_corruption():
    from types import SimpleNamespace
    from hyperell_regulator import frobenius
    with patch.object(frobenius.subprocess, 'run') as run:
        run.side_effect = [SimpleNamespace(stdout=smoke_output()),
                           SimpleNamespace(stdout=smoke_output().replace(' 0', ' 2'))]
        with pytest.raises(RuntimeError, match='parallel matrices differ'):
            frobenius._check_binary('/euler')


def test_accelerator_check_includes_cpu_count():
    from hyperell_regulator import frobenius
    with patch.object(frobenius.os, 'cpu_count', return_value=16), \
            patch.object(frobenius.subprocess, 'run') as run:
        run.return_value.stdout = smoke_output()
        assert frobenius._check_binary('/euler') == (1, 4, 16)
        assert [call.args[0] for call in run.call_args_list] == [
            ['/euler', '1'], ['/euler', '4'], ['/euler', '16']]


@pytest.mark.parametrize('cpus, override, expected', [
    (16, None, 16), (None, None, 1), (16, '3', 3), (16, '0', 1),
])
def test_build_jobs(cpus, override, expected):
    module = build_module()
    env = {} if override is None else {'HYPERELL_REGULATOR_BUILD_JOBS': override}
    with patch.dict(os.environ, env, clear=True), \
            patch.object(module['os'], 'cpu_count', return_value=cpus):
        assert module['_build_jobs']() == expected


@pytest.mark.parametrize('cpus, nthreads, expected', [
    (16, None, 16), (None, None, 1), (16, 3, 3), (16, 0, 1),
])
def test_frobenius_worker_count(cpus, nthreads, expected):
    from hyperell_regulator import frobenius
    with patch.object(frobenius, 'euler_binary', return_value='/euler'), \
            patch.object(frobenius.os, 'cpu_count', return_value=cpus), \
            patch.object(frobenius.subprocess, 'run') as run:
        run.return_value.stdout = '101 1 1 7\n'
        result = frobenius._run([1, 0, 0, 1, 0, 1], [(101, 1)], nthreads)
    assert run.call_args.args[0] == ['/euler', str(expected)]
    assert result == {101: [[7]]}
