"""The machine's CPU limit (mesh_jig.cpu): decided from the environment, never by the model driving the jig, applied at
every entry point and before every Blender build, and inherited by what mesh-jig starts."""
import subprocess
import sys

import pytest

from mesh_jig import build, cli, cpu, evaluate

from test_build import GOOD, SPEC, fake_blender

VR = {cpu.CPUS_ENV: "12-15", cpu.YIELD_ENV: "vrcompositor.exe, vrserver.exe"}


def running(*names):
    return lambda: set(names)


@pytest.mark.parametrize("text, cpus", [("12-15", [12, 13, 14, 15]), ("12,13,14,15", [12, 13, 14, 15]),
                                        ("0-1, 8", [0, 1, 8]), ("3", [3])])
def test_cpu_lists_and_ranges_parse(text, cpus):
    assert cpu.parse_cpus(text) == cpus


@pytest.mark.parametrize("text", ["", "a-b", "15-12", "12-", "-3"])
def test_a_list_that_names_no_cpu_is_refused(text):
    with pytest.raises(ValueError):
        cpu.parse_cpus(text)


def test_the_mask_has_one_bit_a_cpu():
    assert cpu.mask([12, 13, 14, 15]) == 0xF000 and cpu.mask([0]) == 1


def test_no_setting_means_no_limit():
    assert cpu.decide({}, running("vrcompositor.exe"))["limited"] is False


def test_cpus_alone_always_limit():
    d = cpu.decide({cpu.CPUS_ENV: "12-15"}, running(), lambda: 16)
    assert d == {"limited": True, "cpus": [12, 13, 14, 15], "priority": "below_normal", "because": "MESH_JIG_CPUS is set"}


def test_the_limit_follows_whether_vr_is_running():
    assert cpu.decide(VR, running("explorer.exe"), lambda: 16)["limited"] is False
    on = cpu.decide(VR, running("explorer.exe", "vrcompositor.exe"), lambda: 16)
    assert on["limited"] and on["because"] == "vrcompositor.exe is running"


@pytest.mark.parametrize("env", [{cpu.CPUS_ENV: "12-20"}, {cpu.CPUS_ENV: "12-15", cpu.PRIORITY_ENV: "realtime"}])
def test_a_limit_the_machine_cannot_have_is_refused(env):
    with pytest.raises(ValueError):
        cpu.decide(env, running(), lambda: 16)


def test_govern_applies_a_limit_and_restores_the_machine_when_vr_stops():
    applied = []
    on = cpu.govern(VR, running("vrserver.exe"), applied.append)
    off = cpu.govern(VR, running(), applied.append)
    assert on["limited"] and not off["limited"]
    assert [a["cpus"] for a in applied] == [[12, 13, 14, 15], None], "the second call gives back every CPU"
    assert cpu.current() == off


def test_with_nothing_set_govern_touches_nothing():
    applied = []
    cpu.govern({}, running("vrserver.exe"), applied.append)
    assert applied == []


def test_a_bad_setting_or_a_refused_call_never_stops_the_work():
    assert "not applied" in cpu.govern({cpu.CPUS_ENV: "x"}, running(), lambda d: None)["because"]

    def refuse(decision):
        raise OSError(5, "access denied")
    d = cpu.govern({cpu.CPUS_ENV: "0"}, running(), refuse)
    assert not d["limited"] and "not applied" in d["because"]


def test_blender_is_told_how_many_cpus_it_has_before_its_script_runs(tmp_path, monkeypatch):
    script = tmp_path / "a01" / "draft.py"
    script.parent.mkdir()
    script.write_text(GOOD)
    limited = {"limited": True, "cpus": [12, 13, 14, 15], "priority": "below_normal", "because": "test"}
    monkeypatch.setattr(cpu, "govern", lambda: limited)
    calls = []
    assert build.build_variants(script, tmp_path / "a01" / "out", SPEC, blender="b", runner=fake_blender(calls)) == ""
    assert calls[0][:6] == ["b", "--background", "--factory-startup", "--threads", "4", "--python"]


def test_every_cli_command_is_governed_before_it_runs(monkeypatch):
    class Governed(Exception):
        pass

    def stop():
        raise Governed
    monkeypatch.setattr(cpu, "govern", stop)
    with pytest.raises(Governed):
        cli.main(["doctor"])


def test_an_evaluation_records_the_limit_its_build_ran_under(tmp_path, monkeypatch):
    from mesh_jig import project
    proj = project.Project(tmp_path, {"model": {"glb": "m.glb"}})
    (tmp_path / "m.py").write_text("def build():\n    pass\n")
    limited = {"limited": True, "cpus": [12, 13], "priority": "below_normal", "because": "vrserver.exe is running"}

    def build_fn(*a, **kw):
        monkeypatch.setattr(cpu, "_last", limited)          # what govern() leaves behind inside a build
        return "stopped after the build for the test"
    rec = evaluate.evaluate(proj, tmp_path / "m.py", tmp_path / "out", build_fn=build_fn, log=lambda *_: None)
    assert rec["stage"] == "build" and rec["cpu"] == limited


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows calls")
def test_a_real_process_and_its_child_keep_to_the_limit():
    code = ("import ctypes, subprocess, sys\n"
            "from mesh_jig import cpu\n"
            "cpu.apply({'cpus': [0], 'priority': 'below_normal'})\n"
            "child = subprocess.run([sys.executable, '-c', 'import ctypes; k = ctypes.windll.kernel32; '\n"
            "  'k.GetCurrentProcess.restype = ctypes.c_void_p; a, s = ctypes.c_size_t(), ctypes.c_size_t(); '\n"
            "  'k.GetProcessAffinityMask(ctypes.c_void_p(k.GetCurrentProcess()), ctypes.byref(a), ctypes.byref(s)); '\n"
            "  'print(a.value, k.GetPriorityClass(ctypes.c_void_p(k.GetCurrentProcess())))'], capture_output=True, text=True)\n"
            "print(child.stdout.strip())\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["1", str(cpu.PRIORITIES["below_normal"])]
