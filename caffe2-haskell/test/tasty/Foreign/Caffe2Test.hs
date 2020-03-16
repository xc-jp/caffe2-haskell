module Foreign.Caffe2Test(tests) where

import ClassyPrelude

import qualified Data.Vector.Storable as Vector
import Foreign.Caffe2
import Foreign.Caffe2.Caffe2Elt
import Test.Tasty
import Test.Tasty.HUnit (testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Foreign.Caffe2"
    [ testReadAfterInput
    , testGetShape
    ]

testReadAfterInput :: TestTree
testReadAfterInput = testCase "readBlob . writeBlob === identity" $ do
  let tensor = Vector.fromList [1, 2]
  ws <- initWorkspace
  blob <- createBlob ws "tensor"
  writeBlob [2] Caffe2Float tensor blob
  tensor' <- readBlob 2 Caffe2Float blob
  tensor' @?= tensor

testGetShape :: TestTree
testGetShape = testCase "get shape of written blob" $ do
  let tensor = Vector.fromList [1..6]
  ws <- initWorkspace
  blob <- createBlob ws "tensor"
  writeBlob [2,3] Caffe2Float tensor blob
  dims <- getShape blob
  dims @?= [2,3]
