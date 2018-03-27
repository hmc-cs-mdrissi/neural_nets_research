module ForTestSetJsonify where

import ForParser

import System.Environment
import System.Exit
import Data.ByteString.Lazy (writeFile)
import Control.Monad
import Data.Aeson

import Data.Either
import Prelude hiding (writeFile)

testSetDirectoryPath :: String
testSetDirectoryPath = "FORTestSet/"

encodedSetDirectoryPath :: String
encodedSetDirectoryPath = "FORTestSetEncodings/"

testFileName :: String
testFileName = "ForTestSet"

filenamePrefix :: String
filenamePrefix = "prog"

readFileNamed :: String -> IO ProgFor
readFileNamed filename = do contents <- readFile (testSetDirectoryPath ++ filename)
                            case parseProg contents of
                                Right prog -> return prog
                                Left e -> die "Unable to parse For program of file"

-- 40 Prog programs are currently present
readAndEncodeFileOfIndex :: Int -> IO ProgFor
readAndEncodeFileOfIndex index = do indexString <- return (show index)
                                    readFileNamed (filenamePrefix ++ indexString)

encodeFilesFromIndices :: Int -> Int -> IO ()
encodeFilesFromIndices lowerIndex upperIndex = do progs <- sequence (map readAndEncodeFileOfIndex [lowerIndex .. upperIndex])
                                                  writeFile (encodedSetDirectoryPath ++ testFileName ++ ".json") $ encode progs
