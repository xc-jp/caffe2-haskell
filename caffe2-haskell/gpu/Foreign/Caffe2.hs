{-# LANGUAGE CPP                 #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}

module Foreign.Caffe2
  ( -- * 'Workspace'
    Workspace
  , BlobPtr
  , NetworkPtr
  , initWorkspace
  , freeWorkspace
    -- * Blob operations
  , hasBlob
  , getBlob
  , createBlob
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
import           Foreign.C                    (CInt, CSize)
import qualified Language.C.Inline.Cpp        as C

import           Foreign.Caffe2.Caffe2Elt     (Caffe2Elt (..))
import           Foreign.Caffe2.Workspace     hiding (createNetwork,
                                               initWorkspace, runNetOnce)

C.context (C.cppCtx <> C.funCtx <> C.vecCtx <> C.bsCtx)

C.include "<iostream>"
C.include "<caffe2/core/init.h>"
C.include "<caffe2/core/net.h>"
C.include "<caffe2/core/operator.h>"
C.include "<caffe2/core/operator_gradient.h>"
C.include "<caffe2/proto/caffe2.pb.h>"
C.include "<caffe2/core/common_gpu.h>"
C.include "<caffe2/core/context_gpu.h>"

C.using "namespace caffe2"

-- | Initialize 'Workspace'.

initWorkspace :: IO Workspace
initWorkspace =
  Workspace <$> [C.block| void* {
    int argc = 0;
    char **argv = {};
    caffe2::GlobalInit(&argc, &argv);
    DeviceOption option;
    option.set_device_type(PROTO_CUDA);
    new CUDAContext(option);
    Workspace *workspace = new Workspace();
    return workspace;
  } |]

#define READ_TENSOR(BLOB,CTYPE)                                                          \
  Mutable.new (fromIntegral outputSize) >>= \vec ->                                      \
  [C.block| void {                                                                       \
    Blob *blob = static_cast<Blob *>($(void *BLOB));                                     \
    Tensor outTensor = Tensor (blob->Get<Tensor>(),CPU);                                 \
    const CTYPE* tensor = outTensor.data<CTYPE>();                                       \
    for(size_t i = 0; i < $(int outputSize); i++) {                                      \
      $vec-ptr:(CTYPE *vec)[i] = tensor[i];                                              \
    }                                                                                    \
  }|] *> Vector.unsafeFreeze vec

-- | Read tensor data stored in a blob

readBlob
  :: Int                        -- ^ Expected size of tensor
  -> Caffe2Elt e                -- ^ Expected element type
  -> BlobPtr
  -> IO (Vector.Vector e)
readBlob  size elt (BlobPtr blob) =
  case elt of
    Caffe2Float  -> coerce $ READ_TENSOR(blob,float)
    Caffe2Double -> coerce $ READ_TENSOR(blob,double)
    Caffe2Word8  -> coerce $ READ_TENSOR(blob,uint8_t)
    Caffe2Word16 -> coerce $ READ_TENSOR(blob,uint16_t)
    Caffe2Word32 -> do
        sayErr "Warning: Caffe2 doesn't support Word32"
        sayErr "Reading the blob as Int32"
        coerce $ READ_TENSOR(blob,int32_t)
    Caffe2Word64 -> do
        sayErr "Warning: Caffe2 doesn't support Word64"
        sayErr "Reading the blob as Int64"
        coerce $ READ_TENSOR(blob,int64_t)
    Caffe2Int8   -> coerce $ READ_TENSOR(blob,int8_t)
    Caffe2Int16  -> coerce $ READ_TENSOR(blob,int16_t)
    Caffe2Int32  -> coerce $ READ_TENSOR(blob,int32_t)
    Caffe2Int64  -> coerce $ READ_TENSOR(blob,int64_t)
    Caffe2Unit   -> pure $ Vector.replicate size ()
  where
    outputSize = fromIntegral size :: CInt

#define READ_CPU_TENSOR(BLOB,CTYPE)                                                      \
  Mutable.new (fromIntegral outputSize) >>= \vec ->                                      \
  [C.block| void {                                                                       \
    Blob *blob = static_cast<Blob *>($(void *BLOB));                                     \
    const TensorCPU &outTensor = blob->Get<TensorCPU>();                                 \
                                                                                         \
    const CTYPE* tensor = outTensor.data<CTYPE>();                                       \
    for(size_t i = 0; i < $(int outputSize); i++) {                                      \
      $vec-ptr:(CTYPE *vec)[i] = tensor[i];                                              \
    }                                                                                    \
  }|] *> Vector.unsafeFreeze vec

-- | Read tensor data stored in a blob on the CPU

readCPUBlob
  :: Int                        -- ^ Expected size of tensor
  -> Caffe2Elt e                -- ^ Expected element type
  -> BlobPtr                    -- ^ Name of tensor
  -> IO (Vector.Vector e)
readCPUBlob size elt (BlobPtr blob) =
  case elt of
    Caffe2Float  -> coerce $ READ_CPU_TENSOR(blob,float)
    Caffe2Double -> coerce $ READ_CPU_TENSOR(blob,double)
    Caffe2Word8  -> coerce $ READ_CPU_TENSOR(blob,uint8_t)
    Caffe2Word16 -> coerce $ READ_CPU_TENSOR(blob,uint16_t)
    Caffe2Word32 -> do
        sayErr "Warning: Caffe2 doesn't support Word32"
        sayErr "Reading the blob as Int32"
        coerce $ READ_CPU_TENSOR(blob,int32_t)
    Caffe2Word64 -> do
        sayErr "Warning: Caffe2 doesn't support Word64"
        sayErr "Reading the blob as Int64"
        coerce $ READ_CPU_TENSOR(blob,int64_t)
    Caffe2Int8   -> coerce $ READ_CPU_TENSOR(blob,int8_t)
    Caffe2Int16  -> coerce $ READ_CPU_TENSOR(blob,int16_t)
    Caffe2Int32  -> coerce $ READ_CPU_TENSOR(blob,int32_t)
    Caffe2Int64  -> coerce $ READ_CPU_TENSOR(blob,int64_t)
    Caffe2Unit   -> pure $ Vector.replicate size ()
  where
    outputSize = fromIntegral size :: CInt

#define INPUT_TENSOR(BLOB,CTYPE,VEC,SIZE,DIMS)                                    \
    Vector.unsafeWith (VEC) $ \ptr -> [C.block| void {                            \
      int64_t *arrInputDims = $vec-ptr:(int64_t *DIMS);                           \
      std::vector<int64_t> shape =                                                \
        std::vector<int64_t>(arrInputDims, arrInputDims + $vec-len:DIMS);         \
      Blob *blob = static_cast<Blob *>($(void *BLOB));                            \
      /* Intermediate CPU Tensor */                                               \
      Tensor tensor = Tensor(shape, CPU);                                         \
      CTYPE *inputVec = $(CTYPE *ptr);                                            \
      size_t tensorSize = $(size_t SIZE);                                         \
      /* Do not copy the data since it is copied when moved to GPU */             \
      tensor.ShareExternalPointer(inputVec, tensorSize);                          \
      /* Copy CPU tensor to a new GPU tensor (allocated on the heap) */           \
      TensorCUDA *gpuTensor = new Tensor (tensor, CUDA);                          \
      /* Set the blob to point at newly created GPU tensor */                     \
      blob->Reset(gpuTensor);                                                     \
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
      INPUT_TENSOR(blob,float,coerce vec,inputSize,inputDimsVec)
    Caffe2Double ->
      INPUT_TENSOR(blob,double,coerce vec,inputSize,inputDimsVec)
    Caffe2Word8 ->
      INPUT_TENSOR(blob,uint8_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Word16 ->
      INPUT_TENSOR(blob,uint16_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Word32 -> do
       sayErr "Caffe2 doesn't support Word32"
       sayErr "inputting tensor as Int32 instead"
       INPUT_TENSOR(blob,int32_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Word64 -> do
       sayErr "Caffe2 doesn't support Word64"
       sayErr "inputting tensor as Int64 instead"
       INPUT_TENSOR(blob,int64_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Int8 ->
      INPUT_TENSOR(blob,int8_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Int16 ->
      INPUT_TENSOR(blob,int16_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Int32 ->
      INPUT_TENSOR(blob,int32_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Int64 ->
      INPUT_TENSOR(blob,int64_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Unit -> pure ()
  where
    inputSize :: CSize
    inputSize = fromIntegral $ product dims
    inputDimsVec :: Vector.Vector Int64
    inputDimsVec = Vector.fromList (fromIntegral <$> dims)

#define INPUT_CPU_TENSOR(BLOB,CTYPE,VEC,SIZE,DIMS)                                       \
 Vector.unsafeWith (VEC) $ \ptr -> [C.block| void {                                      \
    Blob *blob = static_cast<Blob *>($(void * BLOB));                                    \
    auto *tensor = BlobGetMutableTensor(blob, CPU);                                      \
    int64_t *arrInputDims = $vec-ptr:(int64_t *DIMS);                                    \
    std::vector<int64_t> shape =                                                         \
      std::vector<int64_t>(arrInputDims, arrInputDims + $vec-len:DIMS);                  \
    tensor->Resize(shape);                                                               \
    CTYPE *inputVec = $(CTYPE *ptr);                                                     \
    CTYPE *tensorData = tensor->mutable_data<CTYPE>();                                   \
    int tensorSize = $(int SIZE);                                                        \
    for (int i = 0; i < tensorSize; i ++) {                                              \
      tensorData[i] = inputVec[i];                                                       \
    }                                                                                    \
  }|]

-- Write tensor data to a blob stored on the CPU

writeCPUBlob
  :: [Int]                      -- ^ Input shape
  -> Caffe2Elt e                -- ^ Input element type
  -> Vector.Vector e            -- ^ Input data
  -> BlobPtr
  -> IO ()
writeCPUBlob dims elt vec (BlobPtr blob) =
  case elt of
    Caffe2Float ->
      INPUT_CPU_TENSOR(blob,float,coerce vec,inputSize,inputDimsVec)
    Caffe2Double ->
      INPUT_CPU_TENSOR(blob,double,coerce vec,inputSize,inputDimsVec)
    Caffe2Word8 ->
      INPUT_CPU_TENSOR(blob,uint8_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Word16 ->
      INPUT_CPU_TENSOR(blob,uint16_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Word32 -> do
       sayErr "Caffe2 doesn't support Word32"
       sayErr "inputting tensor as Int32 instead"
       INPUT_CPU_TENSOR(blob,int32_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Word64 -> do
       sayErr "Caffe2 doesn't support Word64"
       sayErr "inputting tensor as Int64 instead"
       INPUT_CPU_TENSOR(blob,int64_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Int8 ->
      INPUT_CPU_TENSOR(blob,int8_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Int16 ->
      INPUT_CPU_TENSOR(blob,int16_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Int32 ->
      INPUT_CPU_TENSOR(blob,int32_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Int64 ->
      INPUT_CPU_TENSOR(blob,int64_t,coerce vec,inputSize,inputDimsVec)
    Caffe2Unit -> pure ()
  where
    inputSize :: CInt
    inputSize = fromIntegral $ product dims
    inputDimsVec :: Vector.Vector Int64
    inputDimsVec = Vector.fromList (fromIntegral <$> dims)

-- | Create a network and add it to the 'Workspace'

createNetwork :: Workspace -> ByteString -> IO NetworkPtr
createNetwork (Workspace workspace) netBytes =
     NetworkPtr <$> [C.block| void * {
        Workspace * workspace = static_cast<Workspace *>($(void * workspace));
        NetDef network;
        std::string netString($bs-ptr:netBytes, $bs-len:netBytes);
        CAFFE_ENFORCE(network.ParseFromString(netString));
        network.mutable_device_option()->set_device_type(PROTO_CUDA);
        NetBase *net = workspace->CreateNet(network);
        CAFFE_ENFORCE(net, "Created network must be non-null");
        return net;
      } |]

-- | Run a network without adding it to the 'Workspace'

runNetOnce :: Workspace -> ByteString -> IO ()
runNetOnce (Workspace workspace) netBytes =
      [C.block| void {
        Workspace * workspace = static_cast<Workspace *>($(void * workspace));
        NetDef network;
        std::string netString($bs-ptr:netBytes, $bs-len:netBytes);
        CAFFE_ENFORCE(network.ParseFromString(netString));
        network.mutable_device_option()->set_device_type(PROTO_CUDA);
        CAFFE_ENFORCE(workspace->RunNetOnce(network));
      } |]
