#!/usr/bin/env bash
stack ghci --flag caffe2-haskell:use-gpu caffe2-haskell --nix-shell-options="--arg cudaSupport true"
