{ withHoogle ? false
, nixpkgs ? builtins.fetchTarball https://github.com/NixOS/nixpkgs/archive/e6deb2955c2b04c02a9bfb92a19cc027b2271b90.tar.gz
, cudaSupport ? false
}@args:

let
  withIntrinsics = stdenv: stdenv //
    { mkDerivation = args: stdenv.mkDerivation (args // {
        NIX_CFLAGS_COMPILE = toString (args.NIX_CFLAGS_COMPILE or "") + " -mavx -mavx2 -mfma";
      });
    };
  pkgs = import nixpkgs {};

  caffe2-pytorch = pkgs.callPackage ./caffe2.nix {
    inherit (pkgs.python3Packages) python future six numpy pydot pyyaml;
    # Use python with the standard version of protobuf.
    python-protobuf = pkgs.python3Packages.protobuf.override { protobuf = pkgs.protobuf; };

    # Images are loaded outside of Caffe2
    useOpencv3 = false;

    # Datasets are loaded outside of Caffe2
    useLmdb = false;

    # Enable cuda support when requested.  Never build with Cudnn support.
    useCudnn = false;

    # Caffe2 has trouble building with older versions of CUDA.
    cudatoolkit = pkgs.cudatoolkit;

    stdenv = withIntrinsics pkgs.stdenv;
  };

in

pkgs.haskell.lib.buildStackProject {
  name = "stack-build-greenia";

  src = ./.;

  buildInputs = [ caffe2-pytorch ] ++ pkgs.lib.optional cudaSupport pkgs.cudatoolkit;

  inherit withHoogle;
}
