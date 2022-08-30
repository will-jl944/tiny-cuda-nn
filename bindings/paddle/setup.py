import os

import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Find version of tinycudann by scraping CMakeLists.txt
with open(os.path.join(ROOT_DIR, "CMakeLists.txt"), "r") as cmakelists:
	for line in cmakelists.readlines():
		if line.strip().startswith("VERSION"):
			VERSION = line.split("VERSION")[-1].strip()
			break

print(f"Building PaddlePaddle extension for tiny-cuda-nn version {VERSION}")

ext_modules = []

if paddle.device.is_compiled_with_cuda():
	include_networks = True
	if "--no-networks" in sys.argv:
		include_networks = False
		sys.argv.remove("--no-networks")
		print("Building >> without << neural networks (just the input encodings)")

	if os.name == "nt":
		def find_cl_path():
			import glob
			for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
				paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
				if paths:
					return paths[0]

		# If cl.exe is not on path, try to find it.
		if os.system("where cl.exe >nul 2>nul") != 0:
			cl_path = find_cl_path()
			if cl_path is None:
				raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
			os.environ["PATH"] += ";" + cl_path

	major, minor = paddle.device.cuda.get_device_capability()
	compute_capability = major * 10 + minor

	nvcc_flags = [
		"-std=c++14",
		"--extended-lambda",
		"--expt-relaxed-constexpr",
		# The following definitions must be undefined
		# since TCNN requires half-precision operation.
		"-U__CUDA_NO_HALF_OPERATORS__",
		"-U__CUDA_NO_HALF_CONVERSIONS__",
		"-U__CUDA_NO_HALF2_OPERATORS__",
		f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
		f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
	]
	if os.name == "posix":
		cflags = ["-std=c++14"]
		nvcc_flags += [
			"-Xcompiler=-mf16c",
			"-Xcompiler=-Wno-float-conversion",
			"-Xcompiler=-fno-strict-aliasing",
		]
	elif os.name == "nt":
		cflags = ["/std:c++14"]

	print(f"Targeting compute capability {compute_capability}")

	definitions = [
		f"-DTCNN_MIN_GPU_ARCH={compute_capability}"
	]
	nvcc_flags += definitions
	cflags += definitions

	# List of sources.
	bindings_dir = os.path.dirname(__file__)
	root_dir = os.path.abspath(os.path.join(bindings_dir, "../.."))
	source_files = [
		"tinycudann/bindings.cpp",
		"../../dependencies/fmt/src/format.cc",
		"../../dependencies/fmt/src/os.cc",
		"../../src/cpp_api.cu",
		"../../src/common.cu",
		"../../src/common_device.cu",
		"../../src/encoding.cu",
	]

	if include_networks:
		source_files += [
			"../../src/network.cu",
			"../../src/cutlass_mlp.cu",
		]

		if compute_capability > 70:
			source_files.append("../../src/fully_fused_mlp.cu")
	else:
		nvcc_flags.append("-DTCNN_NO_NETWORKS")
		cflags.append("-DTCNN_NO_NETWORKS")

	ext = CUDAExtension(
		name="tinycudann_bindings._C",
		sources=source_files,
		include_dirs=[
			"%s/include" % root_dir,
			"%s/dependencies" % root_dir,
			"%s/dependencies/cutlass/include" % root_dir,
			"%s/dependencies/cutlass/tools/util/include" % root_dir,
			"%s/dependencies/fmt/include" % root_dir,
			"%s/dependencies/pybind11_json" % root_dir,
			"%s/dependencies/json" % root_dir,
		],
		extra_compile_args={"cxx": cflags, "nvcc": nvcc_flags},
		libraries=["cuda", "cudadevrt", "cudart_static"],
	)
	ext_modules = [ext]
else:
	raise EnvironmentError("PaddlePaddle CUDA is unavailable. tinycudann requires PaddlePaddle to be installed with the CUDA backend.")

setup(
	name="tinycudann",
	version=VERSION,
	# packages=["tinycudann"],
	install_requires=[],
	include_package_data=True,
	zip_safe=False,
	ext_modules=ext_modules,
)
