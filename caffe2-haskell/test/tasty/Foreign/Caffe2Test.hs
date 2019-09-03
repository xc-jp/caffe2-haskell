module Foreign.Caffe2Test(tests) where

import ClassyPrelude

import Data.Array.Accelerate ((:.) (..), Z (..))
import Foreign.Caffe2
import IX.Data.Elt
import IX.Data.ShapeElt
import IX.Data.Tensor
import Test.Tasty
import Test.Tasty.HUnit (testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Foreign.Caffe2"
    [ testReadAfterInput
    ]

testReadAfterInput :: TestTree
testReadAfterInput = testCase "readTensor . inputTensor" $ do
  let tensor = mkTensorSh (Z :. 2) [1, 2] :: Tensor (Z :. Int) Float
  ws <- initWorkspace
  inputTensor ws "tensor" tensor
  tensor' <- readTensor ws (ShapeElt (Z :. 2) SEltTypeFloat) "tensor"
  tensor' @?= tensor
