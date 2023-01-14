#!/usr/bin/env python3

import argparse
import contextlib
import os
import pathlib
import random
import re
import shutil
import signal
import socket
import subprocess
import sys

general_tests = [
    "tests/python/unittest/test_target_codegen_vulkan.py",
    "tests/python/integration",
    "tests/python/unittest",
    "tests/python/relay",
    # 'vta/tests/python/unittest',
    # 'vta/tests/python/integration',
    # Order-dependent tests
    "--ignore=tests/python/unittest/test_tvm_testing_features.py",
    # Also fails on main, not tested in CI because it requires LLVM+GPU.
    "--ignore=tests/python/unittest/test_target_codegen_llvm.py::test_llvm_gpu_lower_atomic",
]

tests_to_run = [
    # "test_ir.py"
    # "tests/python/relay/strategy/arm_cpu/test_depthwise_conv2d.py:TestDepthwiseConv2d_NHWC_HWOI"
    "tests/python/contrib/test_hexagon/test_meta_bug.py"
    # "tests.py"
    # "tests/python/contrib/test_hexagon/test_2d_physical_buffers.py::TestElementWise::test_execute[nhwc-nhwc-int8-16-32-8x8-nchw-8h8w32c-2d-global.vtcm-hexagon]"
    # "tests/python/contrib/test_hexagon/test_launcher.py::test_add",
    # "../qualcomm-dsp/test_meta_sch_e2e.py",
    # "../qualcomm-dsp/tests/python/test_benchmark.py",
    # "../qualcomm-dsp/tests/python/test_meta_sch_e2e_fp16.py",
    # "../qualcomm-dsp/tests/python/test_meta_sch_e2e_int8.py",
    # "../qualcomm-dsp/run_per_layer_perf_hexagon.py",
    # "tests/python/contrib/test_hexagon/test_cache_read_write.py::test_cache_read_write",
    # "tests/python/contrib/test_hexagon/test_benchmark_maxpool2d.py",
    # "tests/python/unittest/test_tir_te_extern_primfunc.py",
    # "tests/python/unittest/test_meta_schedule_relay_tir_compute.py",
    # "tests/python/unittest/test_te_create_primfunc.py",
    # "tests/python/unittest/test_arith_domain_touched.py",
    # "tests/python/contrib/test_hexagon/test_multi_anchor.py"
    # "tests/python/contrib/test_hexagon/test_autotvm.py"
    # "tests/failing.py"
    # "tests/python/contrib/test_hexagon/test_avg_pool2d_slice.py"
    # "tests/python/contrib/test_hexagon/test_blocked_schedules.py"
    # "tests/python/contrib/test_hexagon/test_software_pipeline.py"
    # "tests/python/contrib/test_hexagon/crashy_mccrashface.py"
    # "tests/python/contrib/test_hexagon/test_vectorization.py"
    # "tests/python/unittest/test_tir_texture_scope.py"
    # "tests/python/contrib/test_adreno/test_tir_texture.py"
    # "tests/python/contrib/test_hexagon/test_thread_pool.py::test_speedup_vectorized"
    # "tests/python/contrib/test_hexagon/unit_tests.py"
    # "tests/python/contrib/test_ethosu/test_replace_conv2d.py"
    # "tests/python/contrib/test_hexagon/test_tir.py"
    # "tests/python/unittest/test_meta_schedule_multi_anchor.py"
    # "tests/python/contrib/test_hexagon/test_2d_physical_buffers.py::TestElementWise::test_execute",
    # "tests/python/contrib/test_hexagon/test_multi_anchor.py",
    #
    # *general_tests,
    #
    # "tests/python/contrib/test_hexagon/test_temp_demo.py::test_demo_elementwise",
    # "tests/python/contrib/test_hexagon/test_temp_demo.py::TestElementWise::test_execute",
]


class TrackerRunner:
    def __init__(self, port, env=None, externally_visible=True):
        self.port = port
        self.env = env
        self.address = "0.0.0.0" if externally_visible else get_local_ip()

        self.proc = None

    def start(self):
        self.stop()

        cmd = [
            "python3",
            "-m",
            "tvm.exec.rpc_tracker",
            "--host",
            self.address,
            "--port",
            str(self.port),
        ]
        self.proc = subprocess.Popen(cmd, env=self.env)

    def stop(self):
        if self.proc is not None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# https://stackoverflow.com/a/166589/2689797
def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]


# https://stackoverflow.com/a/52872579/2689797
def is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def default_tracker_port() -> int:
    try:
        return int(os.environ["TVM_TRACKER_PORT"])
    except (KeyError, ValueError):
        # Default to any non-privileged port, and cross fingers that
        # we don't collide with anybody else.
        # tracker_port = random.randint(1024, 65535)

        # Default to any non-privileged port below the default
        # port_end in tvm.rpc.tracker.

        # TODO: Default to a number of ports to try, rather than the
        # end range.  Better still, maybe have the error handling be
        # done at the parent scope.
        while True:
            port = random.randint(1024, 9199 - 1)
            if not is_port_in_use(port):
                return port


def find_available_server_port() -> int:
    # max_port = 65535

    # Currently, no way to specify the port_end when opening a RPC
    # server, so we need to make sure that we're always below the
    # default port_end of 9199.

    # TODO: Default to a number of ports in tvm.rpc.server instead of
    # a port end.  Better yet, let the parent scope manage error
    # handling.
    max_port = 9199 - 9

    while True:
        port = random.randint(1024, max_port)
        if all(not is_port_in_use(port + i) for i in range(10)):
            return port


def get_connected_android_serial_numbers():
    text = subprocess.check_output(["adb", "devices"], encoding="utf-8")
    lines = text.split("\n")
    return [line.split()[0] for line in lines[1:] if line.strip()]


def generate_env(args, stage=None):
    LD_LIBRARY_PATH = [
        args.hexagon_clang / "lib",
    ]
    if stage == "runtime":
        LD_LIBRARY_PATH.append(args.hexagon_toolchain / "lib" / "iss")

    prev_path = os.environ["PATH"]
    sim_dev_path = args.build_dir / "sim_dev-prefix" / "src" / "sim_dev-build"

    updates = dict(
        LD_LIBRARY_PATH=":".join(str(p) for p in LD_LIBRARY_PATH),
        # TVM_TRACKER_HOST=get_local_ip(),
        # TVM_TRACKER_HOST="0.0.0.0",
        # TVM_TRACKER_HOST="169.254.10.61",
        # TVM_TRACKER_PORT=str(args.tracker_port),
        TVM_TRACKER_HOST="",
        TVM_TRACKER_PORT="",
        TVM_SERVER_PORT=str(args.server_port),
        TVM_LIBRARY_PATH=str(args.build_dir),
        ANDROID_SERIAL_NUMBER=args.android_serial_number,
        HEXAGON_SDK_ROOT=str(args.hexagon_sdk),
        HEXAGON_SDK_PATH=str(args.hexagon_sdk),
        HEXAGON_SHARED_LINK_FLAGS=f"-L{args.build_dir.resolve()}/hexagon_api_output -lhexagon_rpc_sim",
        HEXAGON_TOOLCHAIN=args.hexagon_toolchain,
        PATH=f"{prev_path}:{sim_dev_path}",
    )

    env = os.environ.copy()
    env.update(updates)

    return env


def generate_make_cmd(target=None, num_jobs=None):
    cmd = ["make"]
    if target is not None:
        cmd.append(target)

    if num_jobs is not None:
        cmd.append(f"--jobs={num_jobs}")

    return cmd


def build_tvm(args, with_cpptests=True):
    src_dir = args.tvm_home
    build_dir = args.build_dir

    if args.debug_build:
        build_type = "Debug"
    else:
        build_type = "RelWithDebugInfo"

    commands = []
    commands.append(
        [
            "/opt/cmake-3.22.0-rc1-linux-x86_64/bin/cmake",
            f"-DUSE_LLVM={args.llvm_config}",
            "-DUSE_CPP_RPC=ON",
            f"-DCMAKE_CXX_COMPILER={args.hexagon_clang}/bin/clang++",
            "-DCMAKE_CXX_FLAGS=-stdlib=libc++",
            f"-DUSE_HEXAGON_SDK={args.hexagon_sdk}",
            f"-DUSE_HEXAGON_ARCH={args.hexagon_arch}",
            "-DUSE_HEXAGON=ON",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            # Should find a better way to handle this.  When using a
            # linked library for gtest, need to use the same ABI.
            # Current recommendations are to compile gtest from source
            # as part of the build, may see if that can be done.
            "-DCMAKE_PREFIX_PATH=/home/elunderberg/misc/gtest_pre-cpp11-abi",
            "-DUSE_MICRO=OFF",
            "-DUSE_ETHOSU=ON",
            "-DUSE_HEXAGON_QHL=ON",
            str(src_dir.resolve()),
        ]
    )
    print(2112)
    for line in commands[0]:
        print(line)
    commands.append(generate_make_cmd(num_jobs=args.num_jobs))

    if args.cpp_tests is not None:
        commands.append(generate_make_cmd(target="cpptest", num_jobs=args.num_jobs))

    env = generate_env(args)

    env["GTEST_ROOT"] = "/home/elunderberg/misc/gtest_pre-cpp11-abi"
    env["GTEST_LIB"] = "/home/elunderberg/misc/gtest_pre-cpp11-abi"

    build_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        subprocess.check_call(command, cwd=build_dir, env=env)


def build_hexagon_api(args):
    src_dir = args.tvm_home / "apps" / "hexagon_api"
    build_dir = args.build_dir / "hexagon_api_build"
    output_dir = args.build_dir / "hexagon_api_output"

    if args.debug_build:
        build_type = "Debug"
    else:
        build_type = "RelWithDebugInfo"

    cmake_cmd = [
        "/opt/cmake-3.22.0-rc1-linux-x86_64/bin/cmake",
        # f"-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        f"-DUSE_ANDROID_TOOLCHAIN={args.android_toolchain}",
        f"-DANDROID_PLATFORM={args.android_platform}",
        f"-DANDROID_ABI={args.android_abi}",
        f"-DUSE_HEXAGON_ARCH={args.hexagon_arch}",
        f"-DUSE_HEXAGON_SDK={args.hexagon_sdk}",
        f"-DUSE_HEXAGON=ON",
        f"-DUSE_HEXAGON_TOOLCHAIN={args.hexagon_toolchain}",
        f"-DUSE_OUTPUT_BINARY_DIR={output_dir}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        # f"-DUSE_HEXAGON_GTEST={args.hexagon_sdk}/utils/googletest/gtest",
        str(src_dir.resolve()),
    ]
    print(2112)
    for line in cmake_cmd:
        print(line)
    make_cmd = generate_make_cmd(num_jobs=args.num_jobs)

    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(cmake_cmd, cwd=build_dir)
    subprocess.check_call(make_cmd, cwd=build_dir)


def clean(args):
    shutil.rmtree(args.build_dir)


def strip_color_codes(s):
    return re.sub(r"\x1B\[([0-9]{1,2}(;[0-9]{1,2}){0,2})?[m|K]", "", s)


def check_call_log(log_file: pathlib.Path = None, **popen_kwargs):
    if log_file is None:
        subprocess.check_call(**popen_kwargs)

    else:
        with log_file.open("w") as log_file:
            popen_kwargs["stdout"] = subprocess.PIPE
            popen_kwargs["stderr"] = subprocess.STDOUT
            popen_kwargs["encoding"] = "utf-8"

            proc = subprocess.Popen(**popen_kwargs)
            for line in proc.stdout:
                print(line, end="")
                log_file.write(strip_color_codes(line))

            assert proc.returncode == 0


def run_cpp_tests(args):
    env = generate_env(args, stage="runtime")

    cmd = [
        str(args.build_dir / "cpptest"),
    ]

    if args.cpp_tests:
        gtest_filter = ":".join(args.cpp_tests)
        cmd.append(f"--gtest_filter={gtest_filter}")

    kwargs = dict(
        args=cmd,
        cwd=args.tvm_home,
        env=env,
    )
    check_call_log(log_file=args.log_file, **kwargs)


def run_tests(args):
    pytest_cmd = [
        "python3",
        "-mpytest",
        "--color=yes",
        "--verbose",
        *tests_to_run,
    ]

    # Trying to narrow in on a specific problem
    if args.narrow_in:
        pytest_cmd.extend(["--last-failed", "--exitfirst"])

    # Get an overview for how many tests fail.
    if args.pytest_overview:
        pytest_cmd.extend(["--tb=no", "--show-capture=no"])
    else:
        pytest_cmd.append("--capture=no")

    # Run tests in parallel.
    if args.xdist_jobs is not None:
        pytest_cmd.append(f"--numprocesses={args.xdist_jobs}")

    if args.debugger == "pdb":
        pytest_cmd.append("--pdb")
    elif args.debugger == "gdb" and args.gdb_autostart:
        pytest_cmd = ["gdb", "-ex", "run", "--args", *pytest_cmd]
    elif args.debugger == "gdb" and not args.gdb_autostart:
        pytest_cmd = ["gdb", "--args", *pytest_cmd]

    env = generate_env(args, stage="runtime")

    stack = contextlib.ExitStack()

    with stack:
        kwargs = dict(
            args=pytest_cmd,
            cwd=args.tvm_home,
            env=env,
        )

        # TODO: Start the tracker within the pytest command, similar
        # to RPC tests. PR#10581
        # stack.enter_context(TrackerRunner(port=args.tracker_port, env=env))

        check_call_log(log_file=args.log_file, **kwargs)


def main(args):
    for key, value in args._get_kwargs():
        if callable(value):
            setattr(args, key, value(args))

    if args.debugger:
        assert not args.log_file, "Cannot tee to a log file if a debugger is specified"
        assert not args.pytest_overview, "Cannot hide stdout if a debugger is specified"
        assert not args.xdist_jobs, "Cannot run jobs in parallel if a debugger is specified"

    if args.clean:
        clean(args)
    else:
        if args.run_build:
            build_tvm(args)
        if args.cpp_tests is not None:
            run_cpp_tests(args)
        else:
            build_hexagon_api(args)
            run_tests(args)


def normalized_path(path):
    return pathlib.Path(path).resolve()


def arg_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrapper-pdb",
        action="store_true",
        help="Start a pdb post mortem when this wrapper script has an uncaught exception",
    )

    parser.add_argument(
        "--pdb",
        action="store_const",
        dest="debugger",
        default=None,
        const="pdb",
        help="Start a pdb post mortem on uncaught exception from unit tests",
    )

    parser.add_argument(
        "--gdb",
        action="store_const",
        dest="debugger",
        default=None,
        const="gdb",
        help="Start the unit tests in gdb",
    )

    parser.add_argument(
        "--no-gdb-autostart",
        action="store_false",
        dest="gdb_autostart",
        default=True,
    )
    parser.add_argument(
        "--gdb-autostart",
        action="store_true",
        dest="gdb_autostart",
        default=True,
    )

    parser.add_argument(
        "--log-file",
        default=None,
        type=normalized_path,
    )

    # parser.add_argument(
    #     "--cpp-tests",
    #     action="store_true",
    #     help="Build and run the C++ tests",
    # )

    parser.add_argument(
        "--cpp-tests",
        default=None,
        nargs="*",
        help="Build and run the C++ tests",
    )

    parser.add_argument(
        "--pytest-overview",
        action="store_true",
        help="Run tests showing only success/failure",
    )

    parser.add_argument(
        "--narrow-in",
        action="store_true",
        help="Run tests showing only success/failure",
    )

    parser.add_argument(
        "--xdist-jobs",
        type=int,
        default=lambda args: 16 if args.pytest_overview else None,
        help="The number of pytest-xdist jobs to run in parallel (--numprocesses)",
    )

    try:
        tvm_home = pathlib.Path(os.environ["TVM_HOME"])
    except KeyError:
        tvm_home = pathlib.Path(__file__).parent
    parser.add_argument(
        "--tvm-home",
        type=normalized_path,
        default=tvm_home,
    )

    parser.add_argument(
        "--build-dir",
        default=lambda args: args.tvm_home / "build-hexagon",
        type=normalized_path,
    )

    debug_build_default = lambda args: "debug" in str(args.build_dir)
    parser.add_argument(
        "--debug-build",
        action="store_true",
        dest="debug_build",
        default=debug_build_default,
    )
    parser.add_argument(
        "--release-build",
        action="store_false",
        dest="debug_build",
        default=debug_build_default,
    )

    parser.add_argument(
        "--clean",
        action="store_true",
    )

    parser.add_argument(
        "-j",
        "--num-jobs",
        default=None,
        type=int,
        help="Number of jobs to use when compiling.  If unset, uses whatever is set in MAKEFLAGS",
    )

    parser.add_argument(
        "-p",
        "--tracker-port",
        type=int,
        default=lambda args: default_tracker_port(),
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=lambda args: find_available_server_port(),
        help=(
            "All ports from SERVER_PORT to SERVER_PORT+9 "
            "must be available to listen on.  "
            "If unspecified, defaults to a random unprivileged port "
            "satisfying this condition."
        ),
    )

    parser.add_argument(
        "--android-toolchain",
        type=normalized_path,
        default="/opt/Android/android-ndk-r19c/build/cmake/android.toolchain.cmake",
    )

    valid_serial_numbers = ["simulator", *get_connected_android_serial_numbers()]
    parser.add_argument(
        "--android-serial-number",
        default="7bea87c9",
        choices=valid_serial_numbers,
    )

    parser.add_argument(
        "--sim",
        action="store_const",
        dest="android_serial_number",
        const="simulator",
    )

    parser.add_argument(
        "--android-platform",
        default="android-28",
    )

    parser.add_argument(
        "--android-abi",
        default="arm64-v8a",
    )

    parser.add_argument(
        "--hexagon-clang",
        type=normalized_path,
        default="/home/mhessar/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04",
        # default="/opt/qualcomm/hexagon/llvm-16.0.0-qc/lib",
    )

    parser.add_argument(
        "--llvm-config",
        type=normalized_path,
        default="/opt/qualcomm/hexagon/llvm-14.0.0_2/bin/llvm-config",
        # default="/opt/qualcomm/hexagon/llvm-16.0.0-qc/bin/llvm-config",
    )

    parser.add_argument(
        "--hexagon-sdk",
        type=normalized_path,
        # default="/opt/qualcomm/hexagon/SDK/4.2.0.2",
        default="/opt/qualcomm/hexagon/SDK/4.5.0.3",
    )

    parser.add_argument(
        "--hexagon-toolchain",
        type=normalized_path,
        default="/opt/qualcomm/hexagon/Toolchain/8.5.06.1/Tools",
    )

    parser.add_argument(
        "--hexagon-arch",
        default="v68",
    )

    parser.add_argument(
        "--hexagon-device",
        default="sim",
    )

    parser.add_argument(
        "--run-build",
        default=True,
        action="store_true",
        dest="run_build",
    )

    parser.add_argument(
        "--no-run-build",
        default=True,
        action="store_false",
        dest="run_build",
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception:
        if args.wrapper_pdb:
            import pdb, traceback

            traceback.print_exc()
            pdb.post_mortem()
        raise


if __name__ == "__main__":
    arg_main()
