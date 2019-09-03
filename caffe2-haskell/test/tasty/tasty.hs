module Main (main) where

import ClassyPrelude

import qualified Foreign.Caffe2Test
import Test.Tasty

main :: IO ()
main = defaultMain tests

tests :: TestTree
tests = testGroup "Unit tests"
  [ Foreign.Caffe2Test.tests
  ]
