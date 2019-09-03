{-# LANGUAGE CPP             #-}
{-# LANGUAGE QuasiQuotes     #-}
{-# LANGUAGE TemplateHaskell #-}

module Foreign.Caffe2
  ( -- * 'Workspace'
    Workspace
  , initWorkspace
  , freeWorkspace
  , hasBlob
    -- * Tensor operations
  , initTensors
  , readTensor
  , readCPUTensor
  , readTensors
  , inputTensors
  , inputTensor
  , inputCPUTensor
  -- * Network operations
  , run
  , runNetwork
  , InitMode(..)
  , initNetwork
  )
  where

import ClassyPrelude

import Data.Array.Accelerate (Shape, arraySize)
import Data.Array.Accelerate.Array.Sugar (shapeToList)
import Data.Array.Accelerate.Array.Sugar as Acc
import Data.Array.Accelerate.IO.Data.Vector.Storable (fromVectors)
import Data.Coerce (coerce)
import Data.Constraint (Dict (..))
import Data.ProtoLens (encodeMessage)
import Data.Proxy (Proxy (..))
import qualified Data.Vector.Storable as Vector
import qualified Data.Vector.Storable.Mutable as Mutable
import Foreign.C (CInt)
import Foreign.Ptr (Ptr, castPtr)
import qualified Language.C.Inline.Cpp as C

import IX.Data.Elt (KnownElt (..), SEltType (..))
import IX.Data.ShapeElt (ShapeElt (..))
import IX.Data.ShapeElt.Tree (Elts, ShapeEltTree (..))
import IX.Data.Shapes (SafeShape)
import IX.Data.Sized (Sized (SizeOf))
import IX.Data.Tensor (Tensor (..), tensorShape, tensorSize, toPtrs)
import IX.Data.Tensor.Tree (KnownTensorTree' (..), TensorTree (..),
                            TensorTree' (..), sizedTensorTree, sizedTensorTree')
import IX.Data.Vec (Vec (..), splitVec)
import Proto.Caffe2 (NetDef)

import IX.Data.Tree (KnownTree (..), Tree (..), sizedTree, toTree)


C.context (C.cppCtx <> C.funCtx <> C.vecCtx <> C.bsCtx)

C.include "<iostream>"
C.include "<caffe2/core/init.h>"
C.include "<caffe2/core/net.h>"
C.include "<caffe2/core/operator.h>"
C.include "<caffe2/core/operator_gradient.h>"
C.include "<caffe2/proto/caffe2.pb.h>"
C.include "<google/protobuf/message.h>"
C.include "<google/protobuf/message_lite.h>"
C.include "<google/protobuf/text_format.h>"

C.using "namespace caffe2"

----------------------------------------------------------------------------
-- Workspace

-- | Opaque wrapper for information that is necessary two interact with
-- Caffe2.

newtype Workspace = Workspace (Ptr ())
  deriving (Show)

-- | Initialize 'Workspace'.

initWorkspace :: IO Workspace
initWorkspace =
  Workspace <$> [C.block| void* {
    int argc = 0;
    char **argv = {};
    caffe2::GlobalInit(&argc, &argv);
    Workspace *workspace = new Workspace();
    return workspace;
  } |]

-- | Free 'Workspace'.

freeWorkspace :: Workspace -> IO ()
freeWorkspace (Workspace workspace) =
  [C.block| void {
    Workspace *workspace = static_cast<Workspace *>($(void * workspace));
    for (auto blob : workspace->Blobs()) {
      workspace->RemoveBlob(blob);
    }
    for (auto net : workspace->Nets()) {
      workspace->DeleteNet(net);
    }
  } |]

-- | Verify whether a blob by specified name is present in the 'Workspace'.

hasBlob
  :: Workspace                  -- ^ 'Workspace'
  -> Text                       -- ^ Tensor name
  -> IO Bool
hasBlob (Workspace workspace) name = do
  i <- [C.block| int {
    Workspace * workspace = static_cast<Workspace *>($(void * workspace));
    std::string name($bs-ptr:nameBytes, $bs-len:nameBytes);
    return workspace->HasBlob(name);
  } |]
  pure (i == 1)
    where
    nameBytes = encodeUtf8 name

----------------------------------------------------------------------------
-- Tensor operations

-- | Initialize space for the given tensors.
--

initTensors
  :: (KnownTree Shape is, Sized is)
  => Workspace                 -- ^ 'Workspace'
  -> is                        -- ^ Shapes of tensors
  -> Vec (SizeOf is) Text      -- ^ Names of tensors
  -> IO ()
initTensors workspace shapes =
  go (toTree shapes)
  where
    go :: Tree Shape a -> Vec (SizeOf a) Text -> IO ()
    go (Leaf sh) (name :+ NilV) =
      initTensor workspace sh name
    go (Node a b) names
      | Dict <- sizedTree a
      , Dict <- sizedTree b
      = case splitVec names of
          (as, bs) -> do
            go a as
            go b bs

-- | Initialize a single tensor.

initTensor
  :: Shape sh
  => Workspace                  -- ^ 'Workspace'
  -> sh                         -- ^ Tensor shape
  -> Text                       -- ^ Tensor name
  -> IO ()
initTensor (Workspace workspace) shape' name =
  [C.block| void {
    Workspace * workspace = static_cast<Workspace *>($(void * workspace));

    int64_t *dims = $vec-ptr:(int64_t *dimsVec);
    const std::vector<int64_t> shape = std::vector<int64_t>(dims, dims + $vec-len:dimsVec);

    std::string name($bs-ptr:nameBytes, $bs-len:nameBytes);
    Blob *blob = workspace->CreateBlob(name);
    auto *tensor = BlobGetMutableTensor(blob, CPU);
    tensor->Resize(shape);
  } |]
    where
    dims :: Shape sh => sh -> Vector.Vector Int64
    dims = Vector.fromList . fmap fromIntegral
         . reverse . shapeToList
    dimsVec = dims shape'
    nameBytes = encodeUtf8 name

-- | Read output tensors from Caffe2.

readTensors
  :: (KnownTensorTree' SafeShape KnownElt os)
  => Workspace                         -- ^ 'Workspace'
  -> ShapeEltTree SafeShape KnownElt (Elts os) -- ^ ShapeElts
  -> Vec (SizeOf os) Text              -- ^ Names of outputs
  -> IO os
readTensors workspace shapes =
  go shapes (tensorTree' Proxy)
    where
      go
        :: ShapeEltTree SafeShape KnownElt (Elts a)
        -> TensorTree' SafeShape KnownElt a
        -> Vec (SizeOf a) Text
        -> IO a
      go (ShapeEltLeaf shelt) (TensorLeaf' _) (name :+ NilV) =
        readTensor workspace shelt name
      go (ShapeEltNode a b) (TensorNode' a' b') names
        | Dict <- sizedTensorTree' a'
        , Dict <- sizedTensorTree' b'
        = case splitVec names of
            (as, bs) -> do
              x <- go a a' as
              y <- go b b' bs
              pure (x,y)

#define READ_TENSOR(WS,CTYPE,NAME)                                                       \
  Mutable.new (fromIntegral outputSize) >>= \vec ->                                      \
  [C.block| void {                                                                       \
    Workspace *workspace = static_cast<Workspace *>($(void *WS));                        \
    std::string outName($bs-ptr:NAME,$bs-len:NAME);                                      \
    const TensorCPU &outTensor =                                                         \
      workspace->GetBlob(outName)->Get<TensorCPU>();                                     \
                                                                                         \
    const CTYPE* tensor = outTensor.data<CTYPE>();                                       \
    for(size_t i = 0; i < $(int outputSize); i++) {                                      \
      $vec-ptr:(CTYPE *vec)[i] = tensor[i];                                              \
    }                                                                                    \
  }|] *> Vector.unsafeFreeze vec >>= \vec' ->                                            \
  pure $ Tensor $ fromVectors outShape (coerce vec')

-- | Read a single tensor.

readTensor
  :: (Shape sh, KnownElt e)
  => Workspace                  -- ^ 'Workspace'
  -> ShapeElt sh e              -- ^ Expected shape of tensor
  -> Text                       -- ^ Name of tensor
  -> IO (Tensor sh e)
readTensor (Workspace workspace) (ShapeElt outShape elt) out =
  case elt of
    SEltTypeFloat  -> READ_TENSOR(workspace,float,outBytes)
    SEltTypeDouble -> READ_TENSOR(workspace,double,outBytes)
    SEltTypeWord8  -> READ_TENSOR(workspace,uint8_t,outBytes)
    SEltTypeWord16 -> READ_TENSOR(workspace,uint16_t,outBytes)
    SEltTypeWord32 -> do
        sayErr "Warning: Caffe2 doesn't support Word32"
        sayErr "Reading the blob as Int32"
        READ_TENSOR(workspace,int32_t,outBytes)
    SEltTypeWord64 -> do
        sayErr "Warning: Caffe2 doesn't support Word64"
        sayErr "Reading the blob as Int64"
        READ_TENSOR(workspace,int64_t,outBytes)
    SEltTypeInt8   -> READ_TENSOR(workspace,int8_t,outBytes)
    SEltTypeInt16  -> READ_TENSOR(workspace,int16_t,outBytes)
    SEltTypeInt32  -> READ_TENSOR(workspace,int32_t,outBytes)
    SEltTypeInt64  -> READ_TENSOR(workspace,int64_t,outBytes)
    SEltTypeUnit   -> pure $ Tensor $ Acc.fromList outShape (replicate (arraySize outShape) ())
  where
    outputSize = fromIntegral $ arraySize outShape :: CInt
    outBytes = encodeUtf8 out

-- | Read a single tensor from the CPU workspace.

readCPUTensor
  :: (Shape sh, KnownElt e)
  => Workspace                  -- ^ 'Workspace'
  -> ShapeElt sh e              -- ^ Expected shape of tensor
  -> Text                       -- ^ Name of tensor
  -> IO (Tensor sh e)
readCPUTensor = readTensor

-- | Input tensors.

inputTensors
  :: Workspace                  -- ^ 'Workspace'
  -> TensorTree SafeShape KnownElt is -- ^ Actual tensors to load
  -> Vec (SizeOf is) Text       -- ^ Names of tensors
  -> IO ()
inputTensors workspace =
  go
    where
      go :: TensorTree SafeShape KnownElt is -> Vec (SizeOf is) Text -> IO ()
      go (TensorLeaf t) (name :+ NilV) =
        inputTensor workspace name t
      go (TensorNode a b) names
        | Dict <- sizedTensorTree a
        , Dict <- sizedTensorTree b
        , (as,bs) <- splitVec names
        = go a as *> go b bs

#define INPUT_TENSOR(WS,CTYPE,NAME,PTR,SIZE,DIMS)                                        \
 [C.block| void {                                                                        \
    Workspace *workspace = static_cast<Workspace *>($(void *WS));                        \
    int64_t *arrInputDims = $vec-ptr:(int64_t *DIMS);                                    \
    std::vector<int64_t> shape =                                                         \
      std::vector<int64_t>(arrInputDims, arrInputDims + $vec-len:DIMS);                  \
                                                                                         \
    std::string inputName($bs-ptr:NAME, $bs-len:NAME);                                   \
    Blob *blob = workspace->CreateBlob(inputName);                                       \
    auto *tensor = BlobGetMutableTensor(blob, CPU);                                      \
    tensor->Resize(shape);                                                               \
    CTYPE *inputVec = $(CTYPE *PTR);                                                     \
    CTYPE *tensorData = tensor->mutable_data<CTYPE>();                                   \
    int tensorSize = $(int SIZE);                                                        \
    for (int i = 0; i < tensorSize; i ++) {                                              \
      tensorData[i] = inputVec[i];                                                       \
    }                                                                                    \
  }|]
--
-- | Load a single tensor.

inputTensor
  :: forall sh e. (Shape sh, KnownElt e)
  => Workspace                  -- ^ 'Workspace'
  -> Text                       -- ^ Input name
  -> Tensor sh e            -- ^ Tensor to load
  -> IO ()
inputTensor (Workspace workspace) i tensor =
  case selt (Proxy :: Proxy e) of
    SEltTypeFloat ->
      let ptr = castPtr (toPtrs tensor)
       in INPUT_TENSOR(workspace,float,inputBytes,ptr,inputSize,inputDimsVec)
    SEltTypeDouble ->
      let ptr = castPtr (toPtrs tensor)
       in INPUT_TENSOR(workspace,double,inputBytes,ptr,inputSize,inputDimsVec)
    SEltTypeWord8 ->
      let ptr = castPtr (toPtrs tensor)
       in INPUT_TENSOR(workspace,uint8_t,inputBytes,ptr,inputSize,inputDimsVec)
    SEltTypeWord16 ->
      let ptr = castPtr (toPtrs tensor)
       in INPUT_TENSOR(workspace,uint16_t,inputBytes,ptr,inputSize,inputDimsVec)
    SEltTypeWord32 ->
      let ptr = castPtr (toPtrs tensor)
       in do
       sayErr "Caffe2 doesn't support Word32"
       sayErr "inputting tensor as Int32 instead"
       INPUT_TENSOR(workspace,int32_t,inputBytes,ptr,inputSize,inputDimsVec)
    SEltTypeWord64 ->
      let ptr = castPtr (toPtrs tensor)
       in do
       sayErr "Caffe2 doesn't support Word64"
       sayErr "inputting tensor as Int64 instead"
       INPUT_TENSOR(workspace,int64_t,inputBytes,ptr,inputSize,inputDimsVec)
    SEltTypeInt8 ->
      let ptr = castPtr (toPtrs tensor)
       in INPUT_TENSOR(workspace,int8_t,inputBytes,ptr,inputSize,inputDimsVec)
    SEltTypeInt16 ->
      let ptr = castPtr (toPtrs tensor)
       in INPUT_TENSOR(workspace,int16_t,inputBytes,ptr,inputSize,inputDimsVec)
    SEltTypeInt32 ->
      let ptr = castPtr (toPtrs tensor)
       in INPUT_TENSOR(workspace,int32_t,inputBytes,ptr,inputSize,inputDimsVec)
    SEltTypeInt64 ->
      let ptr = castPtr (toPtrs tensor)
       in INPUT_TENSOR(workspace,int64_t,inputBytes,ptr,inputSize,inputDimsVec)
    SEltTypeUnit -> pure ()
  where
    inputBytes = encodeUtf8 i
    inputSize :: CInt
    inputSize = fromIntegral $ tensorSize tensor
    dims :: Shape sh => Tensor sh a -> Vector.Vector Int64
    dims = Vector.fromList . fmap fromIntegral
         . reverse . shapeToList . tensorShape
    inputDimsVec = dims tensor

-- | Load a single tensor.

inputCPUTensor
  :: forall sh e. (Shape sh, KnownElt e)
  => Workspace                  -- ^ 'Workspace'
  -> Text                       -- ^ Input name
  -> Tensor sh e            -- ^ Tensor to load
  -> IO ()
inputCPUTensor = inputTensor

----------------------------------------------------------------------------
-- Network operations

-- | Network initialization mode.

data InitMode = RunOnce | CreateOnly
  deriving (Eq, Show)

-- | Initialize network.

initNetwork
  :: InitMode                   -- ^ Initialization mode
  -> NetDef                     -- ^ Network definition
  -> Workspace                  -- ^ 'Workspace'
  -> IO ()
initNetwork mode netDef (Workspace workspace) =
  case mode of
    RunOnce ->
      [C.block| void {
        Workspace * workspace = static_cast<Workspace *>($(void * workspace));
        NetDef network;
        std::string netString($bs-ptr:netBytes, $bs-len:netBytes);
        CAFFE_ENFORCE(network.ParseFromString(netString));
        CAFFE_ENFORCE(workspace->RunNetOnce(network));
      } |]
    CreateOnly ->
      [C.block| void {
        Workspace * workspace = static_cast<Workspace *>($(void * workspace));
        NetDef network;
        std::string netString($bs-ptr:netBytes, $bs-len:netBytes);
        CAFFE_ENFORCE(network.ParseFromString(netString));
        CAFFE_ENFORCE(workspace->CreateNet(network));
      } |]
  where
    netBytes = encodeMessage netDef

-- | Run learning or evaluation process.

run
  :: (KnownTensorTree' SafeShape KnownElt os)
  => Text                       -- ^ Network name
  -> Vec (SizeOf is) Text    -- ^ Input names
  -> Vec (SizeOf os) Text    -- ^ Output names
  -> ShapeEltTree SafeShape KnownElt (Elts os)
  -> Workspace                  -- ^ 'Workspace'
  -> TensorTree SafeShape KnownElt is           -- ^ Input tensors
  -> IO os      -- ^ Results
run netName inputs outputs outputShapes workspace samples = do
  inputTensors workspace samples inputs
  runNetwork workspace netName
  readTensors workspace outputShapes outputs

-- | Run network.

runNetwork
  :: Workspace                  -- ^ 'Workspace'
  -> Text                       -- ^ Network name
  -> IO ()
runNetwork (Workspace workspace) netName =
  [C.block| void {
    {
      Workspace *workspace = static_cast<Workspace *>($(void * workspace));
      std::string name($bs-ptr:netBytes,$bs-len:netBytes);
      NetBase *net = workspace->GetNet(name);
      CAFFE_ENFORCE(workspace->RunNet(name));
    }
  }|]
  where
    netBytes = encodeUtf8 netName
