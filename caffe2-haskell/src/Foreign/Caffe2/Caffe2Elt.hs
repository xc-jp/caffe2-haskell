module Foreign.Caffe2.Caffe2Elt (Caffe2Elt(..)) where

import Prelude

import Data.Int
import Data.Word

data Caffe2Elt a where
    Caffe2Float   :: Caffe2Elt Float
    Caffe2Double  :: Caffe2Elt Double
    Caffe2Word8   :: Caffe2Elt Word8
    Caffe2Word16  :: Caffe2Elt Word16
    Caffe2Word32  :: Caffe2Elt Word32
    Caffe2Word64  :: Caffe2Elt Word64
    Caffe2Int8    :: Caffe2Elt Int8
    Caffe2Int16   :: Caffe2Elt Int16
    Caffe2Int32   :: Caffe2Elt Int32
    Caffe2Int64   :: Caffe2Elt Int64
    Caffe2Unit    :: Caffe2Elt ()
