# This derivation is for caffe2.
#
# This installs a version of caffe2 from the pytorch repository (instead of the
# old caffe2 repository).

{ stdenv, lib, config, fetchFromGitHub, git
, cmake
, glog, google-gflags, gtest
, protobuf, snappy
, python, future, six, python-protobuf, numpy, pydot
, pyyaml
, eigen
, doxygen
, useCuda ? (config.cudaSupport or false), cudatoolkit ? null
, useCudnn ? (config.cudnnSupport or false), cudnn ? null
, useOpenmp ? false, openmp ? null
, useOpencv3 ? true, opencv3 ? null
, useLeveldb ? false, leveldb ? null
, useLmdb ? true, lmdb ? null
, useRocksdb ? false, rocksdb ? null
, useZeromq ? false, zeromq ? null
, useMpi ? false, mpi ? null
# TODO: distributed computations
#, useGloo ? false
#, useNccl ? false
#, useNnpack ? false
}:

assert useCuda -> cudatoolkit != null;
assert useCudnn -> (useCuda && cudnn != null);
assert useOpencv3 -> opencv3 != null;
assert useLeveldb -> leveldb != null;
assert useLmdb -> lmdb != null;
assert useRocksdb -> rocksdb != null;
assert useZeromq -> zeromq != null;
assert useMpi -> mpi != null;

stdenv.mkDerivation rec {
  name = "caffe2-${version}";
  version = "1.3.0";
  src = fetchFromGitHub {
    owner = "pytorch";
    repo = "pytorch";
    rev = "v${version}";
    sha256 = "1219m9mfadnif43bd3f64csa3qbx4za7rc0ic6yawgwmx8f6jqn0";
    fetchSubmodules = true;
  };

  nativeBuildInputs = [ cmake doxygen git gtest python pyyaml ];

  buildInputs = [ glog google-gflags protobuf snappy eigen ]
    ++ lib.optional useCuda cudatoolkit
    ++ lib.optional useCudnn cudnn
    ++ lib.optional useOpencv3 opencv3
    ++ lib.optional useLeveldb leveldb
    ++ lib.optional useLmdb lmdb
    ++ lib.optional useRocksdb rocksdb
    ++ lib.optional useZeromq zeromq
  ;
  propagatedBuildInputs = [
    eigen
    glog
    google-gflags
    protobuf
    openmp
  ] ++ lib.optional useCuda cudatoolkit;

  preConfigure = ''
      export CFLAGS="-I ${eigen}/include/eigen3/"
  '';

  cmakeFlags = [ ''-DBUILD_TEST=OFF''
                 ''-DBUILD_PYTHON=OFF''
                 ''-DUSE_CUDA=${if useCuda then ''ON''else ''OFF''}''
                 ''-DUSE_CUDNN=${if useCudnn then ''ON''else ''OFF''}''
                 ''-DUSE_OPENCV=${if useOpencv3 then ''ON''else ''OFF''}''
                 ''-DUSE_MPI=${if useMpi then ''ON''else ''OFF''}''
                 ''-DUSE_LEVELDB=${if useLeveldb then ''ON''else ''OFF''}''
                 ''-DUSE_LMDB=${if useLmdb then ''ON''else ''OFF''}''
                 ''-DUSE_ROCKSDB=${if useRocksdb then ''ON''else ''OFF''}''
                 ''-DUSE_ZMQ=${if useZeromq  then ''ON''else ''OFF''}''
                 ''-DUSE_GLOO=OFF''
                 ''-DUSE_NATIVE_ARCH=ON''
                 ''-DUSE_AVX=ON''
                 ''-DUSE_AVX2=ON''
                 ''-DUSE_FMA=ON''
                 # If you set USE_OPENMP to OFF, libtorch.so still uses symbols
                 # from libgomp.so.  However, it doesn't add libgomp.so as a
                 # NEEDED library to the resulting libtorch.so.  You can
                 # confirm this by running ldd on libtorch.so and seeing that
                 # libgomp.so is not included in this list.
                 #
                 # We set this to ON so that libtorch.so will be correctly
                 # linked to libgomp.so.
                 ''-DUSE_OPENMP=ON''
                 # There is a problem with turning off both python support and NNPACK at the same time.
                 # https://github.com/pytorch/pytorch/issues/17525
                 # ''-DUSE_NNPACK=OFF''
                 ''-DUSE_NCCL=OFF''
                 ''-DUSE_REDIS=OFF''
                 ''-DUSE_FFMPEG=OFF''
                 ''-DBUILD_CUSTOM_PROTOBUF=OFF''
                 ''-DBUILD_DOCS=ON''
               ]
               ++ lib.optional useCuda [
                 ''-DCUDA_TOOLKIT_ROOT_DIR=${cudatoolkit}''
                 ''-DCUDA_FAST_MATH=ON''
                 ''-DCUDA_HOST_COMPILER=${cudatoolkit.cc}/bin/gcc''
               ];

  doCheck = false;
  enableParallelBuilding = true;

  meta = {
    homepage = https://caffe2.ai/;
    description = "A new lightweight, modular, and scalable deep learning framework";
    longDescription = ''
      Caffe2 aims to provide an easy and straightforward way for you to experiment
      with deep learning and leverage community contributions of new models and
      algorithms. You can bring your creations to scale using the power of GPUs in the
      cloud or to the masses on mobile with Caffe2's cross-platform libraries.
    '';
    platforms = with stdenv.lib.platforms; linux ++ darwin;
    license = stdenv.lib.licenses.asl20;
    maintainers = with stdenv.lib.maintainers; [ yuriaisaka ];
  };
}
