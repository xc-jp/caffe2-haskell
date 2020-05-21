{-# LANGUAGE CPP             #-}
{-# LANGUAGE QuasiQuotes     #-}
{-# LANGUAGE TemplateHaskell #-}

module Foreign.Caffe2
  ( -- * 'Workspace'
    Workspace
  , BlobPtr
  , NetworkPtr
  , initWorkspace
  , freeWorkspace
    -- * Blob operations
  , hasBlob
  , createBlob
  , getBlob
  , readBlob
  , readCPUBlob
  , writeBlob
  , writeCPUBlob
  -- * Network operations
  , createNetwork
  , runNetOnce
  , runNetwork
  , getNetwork
  , getShape
  )
  where

import           ClassyPrelude                hiding (Vector)

import           Data.Coerce                  (coerce)
import qualified Data.Vector.Storable         as Vector
import qualified Data.Vector.Storable.Mutable as Mutable
import           Foreign.C                    (CInt)
import qualified Language.C.Inline.Cpp        as C

import           Foreign.Caffe2.Caffe2Elt     (Caffe2Elt (..))
import           Foreign.Caffe2.Workspace

C.context (C.cppCtx <> C.funCtx <> C.vecCtx <> C.bsCtx)

C.include "<iostream>"
C.include "<caffe2/core/init.h>"
C.include "<caffe2/core/net.h>"
C.include "<caffe2/core/operator.h>"
C.include "<caffe2/core/operator_gradient.h>"
C.include "<caffe2/ideep/ideep_utils.h>"
C.include "<caffe2/proto/caffe2.pb.h>"

C.using "namespace caffe2"


#define READ_TENSOR(BLOB,CTYPE,NAME)                                                     \
  Mutable.new (fromIntegral outputSize) >>= \vec ->                                      \
  [C.block| void {                                                                       \
    printf("Reading blob CTYPE NAME ... \n");                                            \
    Blob *blob = static_cast<Blob *>($(void *BLOB));                                     \
    const TensorCPU &outTensor = blob->Get<TensorCPU>();                                 \
                                                                                         \
    const CTYPE* tensor = outTensor.data<CTYPE>();                                       \
    for(size_t i = 0; i < $(int outputSize); i++) {                                      \
      $vec-ptr:(CTYPE *vec)[i] = tensor[i];                                              \
    }                                                                                    \
  }|] *> Vector.unsafeFreeze vec

-- | Read the tensor data stored in a blob

readBlob
  :: Int
  -> Caffe2Elt e
  -> BlobPtr
  -> IO (Vector.Vector e)
readBlob size elt (BlobPtr blob) =
  case elt of
    Caffe2Float  -> coerce $ READ_TENSOR(blob,float,outBytes)
    Caffe2Double -> coerce $ READ_TENSOR(blob,double,outBytes)
    Caffe2Word8  -> coerce $ READ_TENSOR(blob,uint8_t,outBytes)
    Caffe2Word16 -> coerce $ READ_TENSOR(blob,uint16_t,outBytes)
    Caffe2Word32 -> do
        sayErr "Warning: Caffe2 doesn't support Word32"
        sayErr "Reading the blob as Int32"
        coerce $ READ_TENSOR(blob,int32_t,outBytes)
    Caffe2Word64 -> do
        sayErr "Warning: Caffe2 doesn't support Word64"
        sayErr "Reading the blob as Int64"
        fmap coerce $ READ_TENSOR(blob,int64_t,outBytes)
    Caffe2Int8   -> coerce $ READ_TENSOR(blob,int8_t,outBytes)
    Caffe2Int16  -> coerce $ READ_TENSOR(blob,int16_t,outBytes)
    Caffe2Int32  -> coerce $ READ_TENSOR(blob,int32_t,outBytes)
    Caffe2Int64  -> coerce $ READ_TENSOR(blob,int64_t,outBytes)
    Caffe2Unit   -> pure $ Vector.replicate size ()
  where
    outputSize = fromIntegral size :: CInt

-- | Read the tensor stored in a blob on the CPU

readCPUBlob :: Int -> Caffe2Elt e -> BlobPtr -> IO (Vector.Vector e)
readCPUBlob = readBlob
-- | Load a single tensor.
--
#define WRITE_TENSOR(BLOB,CTYPE,VEC,SIZE,DIMS)                                           \
 Vector.unsafeWith (VEC) $ \ptr -> [C.block| void {                                      \
    Blob *blob = static_cast<Blob *>($(void *BLOB));                                     \
    int *arrInputDims = $vec-ptr:(int *DIMS);                                    \
    std::vector<int> shape =                                                         \
      std::vector<int>(arrInputDims, arrInputDims + $vec-len:DIMS);                  \
    auto *tensor = blob->GetMutable<ideep::tensor>();                                    \
    tensor->resize(shape, ideep::tensor::data_type::f32);                                                               \
    CTYPE *inputVec = $(CTYPE *ptr);                                                     \
    // CTYPE *tensorData = tensor->mutable_data<CTYPE>();                                   \
    // int tensorSize = $(int SIZE);                                                        \
    // for (int i = 0; i < tensorSize; i ++) {                                              \
    //   tensorData[i] = inputVec[i];                                                       \
    // }                                                                                    \
    tensor->feed_from(shape, ideep::tensor::data_type::f32, inputVec);                                           \
  }|]

-- | Write tensor data to a blob

writeBlob
  :: [Int]                      -- ^ Input shape
  -> Caffe2Elt e                -- ^ Input element type
  -> Vector.Vector e            -- ^ Input data
  -> BlobPtr
  -> IO ()
writeBlob dims elt vec (BlobPtr blob) =
  case elt of
    Caffe2Float ->
      WRITE_TENSOR(blob,float,coerce vec,inputSize,inputDimsVec)
    Caffe2Double -> do
      sayErr "ideep doesn't support Double"
      -- WRITE_TENSOR(blob,double,coerce vec,inputSize,inputDimsVec)
      pure ()
    Caffe2Word8 -> do
      sayErr "ideep doesn't support Double"
      -- WRITE_TENSOR(blob,uint8_t,coerce vec,inputSize,inputDimsVec)
      pure ()
    Caffe2Word16 -> do
      sayErr "ideep doesn't support Double"
      -- WRITE_TENSOR(blob,uint16_t,coerce vec,inputSize,inputDimsVec)
      pure ()
    Caffe2Word32 -> do
      sayErr "Caffe2 doesn't support Word32"
      sayErr "inputting tensor as Int32 instead"
       -- WRITE_TENSOR(blob,int32_t,coerce vec,inputSize,inputDimsVec)
      pure ()
    Caffe2Word64 -> do
      sayErr "Caffe2 doesn't support Word64"
      sayErr "inputting tensor as Int64 instead"
       -- WRITE_TENSOR(blob,int64_t,coerce vec,inputSize,inputDimsVec)
      pure ()
    Caffe2Int8 -> do
      sayErr "ideep doesn't support Double"
      -- WRITE_TENSOR(blob,int8_t,coerce vec,inputSize,inputDimsVec)
      pure ()
    Caffe2Int16 -> do
      sayErr "ideep doesn't support Double"
      -- WRITE_TENSOR(blob,int16_t,coerce vec,inputSize,inputDimsVec)
      pure ()
    Caffe2Int32 -> do
      sayErr "ideep doesn't support Double"
      -- WRITE_TENSOR(blob,int32_t,coerce vec,inputSize,inputDimsVec)
      pure ()
    Caffe2Int64 -> do
      sayErr "ideep doesn't support Double"
      -- WRITE_TENSOR(blob,int64_t,coerce vec,inputSize,inputDimsVec)
      pure ()
    Caffe2Unit -> pure ()
  where
    inputSize :: CInt
    inputSize = fromIntegral $ product dims
    inputDimsVec :: Vector.Vector CInt
    inputDimsVec = Vector.fromList (fromIntegral <$> dims)

-- | Write tensor data to a blob stored on the CPU

writeCPUBlob :: [Int] -> Caffe2Elt e -> Vector.Vector e -> BlobPtr -> IO ()
writeCPUBlob = writeBlob
