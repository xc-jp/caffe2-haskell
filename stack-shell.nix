# This derivation can be used to create an nix-shell environment for developing
# the Greenia Haskell packages.
#
# This derivation is used from the `stack` wrapper in the `tools` directory.

{ withHoogle ? false
, nixpkgs ? builtins.fetchTarball https://github.com/NixOS/nixpkgs/archive/e6deb2955c2b04c02a9bfb92a19cc027b2271b90.tar.gz
}@args:

let
  withIntrinsics = stdenv: stdenv //
    { mkDerivation = args: stdenv.mkDerivation (args // {
        NIX_CFLAGS_COMPILE = toString (args.NIX_CFLAGS_COMPILE or "") + " -mavx -mavx2 -mfma";
      });
    };
  pkgs = import nixpkgs {};

  # We need an older version (< 2.1) of stack due to issue:
  # https://github.com/commercialhaskell/stack/issues/5000
  inherit (pkgs) writeScriptBin;

  # Stack needs the security binary to be available in the environment to do
  # TLS on macOS
  securityWrapper = writeScriptBin "security" ''
    #!/bin/sh -e
    exec /usr/bin/security "$@"
  '';
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

  # inherit stack;

  buildInputs = with pkgs;
    [
      caffe2-pytorch
      eigen
      git
      glog
      google-gflags
      libffi
      # llvm-config
      postgresql
      protobuf
      # hsPkgs.ShellCheck.components.exes.shellcheck
      # shunit2
      sqlite
      zlib
    ]; # ++
    # lib.optional stdenv.isDarwin securityWrapper ++
    # lib.optional cudaSupport cudatoolkit;

  inherit withHoogle;

  extraArgs = with pkgs;
      ["--extra-lib-dirs=${eigen}/lib" 
       "--extra-include-dirs=${eigen}/include" 
      ];
}
